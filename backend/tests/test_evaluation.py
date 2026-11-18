import copy
import unittest
import httpx
from datetime import datetime, timezone
from types import SimpleNamespace

from pydantic import ValidationError

from qa.api import run_summary
from qa.contracts import Evidence, ModelAnswer, OutcomeAnswer, ReviewInput, Scope, Ticket
from qa.evaluation import assessment_metrics, code_check, coverage_metrics, evaluate, model_check, refund_check, scoped_messages
from qa.outcomes import evaluate_outcome, evaluate_request_outcomes
from qa.settings import CARD

LEGACY_FIRST_RESPONSE = {
    'id': 'first_response', 'version': 2, 'group_id': 'handling', 'name': 'First response time',
    'evaluator_type': 'code', 'weight': 5, 'severity': 'standard', 'review_policy': 'on_fail',
    'channel_threshold_seconds': {'chat': 900},
}


def ticket(**changes):
    data = {
        'external_id': 'test', 'subject': 'Payment refund', 'category': 'refund',
        'messages_complete': True, 'events_complete': True,
        'messages': [
            {'id': 'm1', 'actor_id': 'customer', 'author': 'Customer', 'role': 'customer', 'at': '2026-11-12T09:00:00Z', 'text': 'Please refund my payment.'},
            {'id': 'm2', 'actor_id': 'agent', 'author': 'Agent', 'role': 'agent', 'at': '2026-11-12T09:10:00Z', 'text': 'The refund was issued.'},
        ],
    }
    return Ticket.model_validate({**data, **changes})


def event(kind='refund_completed', **changes):
    return {'id': 'e1', 'actor_id': 'system', 'kind': kind, 'at': '2026-11-12T09:09:00Z', 'detail': 'The transaction was confirmed.', **changes}


def criterion(name):
    if name == 'first_response':
        return LEGACY_FIRST_RESPONSE
    return next(item for item in CARD['criteria'] if item['id'] == name)


class Provider:
    def __init__(self, verdict='pass', outcome='unknown', evidence=None):
        self.verdict, self.outcome, self.evidence = verdict, outcome, evidence
        self.requests = []

    def assess(self, request):
        self.requests.append(request)
        source = next(s for s in request['sources'] if s['type'] == 'message' and s['ref'] in request['target_messages'])
        evidence = self.evidence if self.evidence is not None else [Evidence(type='message', ref=source['ref'], quote=source['text'])]
        return ModelAnswer(verdict=self.verdict, explanation='Evidence for the check.', evidence=evidence), {'eval_count': 5}

    def assess_outcome(self, request):
        self.requests.append(request)
        return OutcomeAnswer(status=self.outcome, explanation='Outcome based on the available history.', evidence=self.evidence or []), {'eval_count': 3}


class RequestOutcomeTests(unittest.TestCase):
    def test_scores_and_coverage_use_only_supported_requests_and_keep_zero(self):
        example = ticket()
        data = example.model_dump(mode='json')
        data['messages'][0]['text'] = 'Refund both charges, restore access, check the delivery and send an invoice.'
        data['messages'][1]['text'] = 'Try the account recovery link to restore access.'
        data['messages'] += [{
            **data['messages'][0], 'id': 'm3', 'at': '2026-11-12T09:15:00Z',
            'text': 'One charge was refunded, the parcel is still missing.',
        }]
        model_rows = [
            ('Refund both charges', 'partly_resolved', 'm3'),
            ('Restore access', 'remedy_proposed', 'm2'),
            ('Check the missing parcel', 'unresolved', 'm3'),
            ('Send an invoice', 'resolved', 'm2'),  # Agent reply cannot establish resolution.
        ]
        class RequestProvider(Provider):
            def extract_outcome_requests(self, request):
                assert all(source['role'] == 'customer' for source in request['sources'])
                return {'requests': [{'request': label, 'request_ref': 'm1'} for label, _, _ in model_rows]
                        + [{'request': 'Confirm contact details', 'request_ref': 'm1'}]}, {'eval_count': 2}

            def assess_request_outcomes(self, request):
                sources = {source['ref']: source['text'] for source in request['sources']}
                return {'requests': [
                    {'request_id': i, 'state': state, 'evidence': [{'type': 'message', 'ref': ref, 'quote': sources[ref]}]}
                    for i, (_, state, ref) in enumerate(model_rows, 1)
                ]}, {'eval_count': 7}

        outcome, usage = evaluate_request_outcomes(Ticket.model_validate(data), RequestProvider())
        self.assertEqual([r['score'] for r in outcome['requests']], [50, 25, 0, None, None])
        self.assertEqual((outcome['score'], outcome['coverage'], outcome['scored'], outcome['total']), (25, 60, 3, 5))
        self.assertTrue(outcome['review_required'])
        self.assertEqual(outcome['requests'][0]['evidence'][0]['ref'], 'm3')
        self.assertEqual(usage, {'eval_count': 9})
        config = {'scope': Scope().model_dump(), 'knowledge': [], 'card': {'criteria': []}}
        self.assertNotIn('request_outcome', evaluate(data, config, RequestProvider()))
        result = evaluate(data, {**config, 'request_outcomes_version': 1}, RequestProvider())
        self.assertEqual(result['request_outcome'], outcome)
        self.assertIsNone(result['score'])


class RefundTests(unittest.TestCase):
    def check(self, expected, scope=None, **changes):
        result = refund_check(ticket(**changes), scope or Scope())
        self.assertEqual(result.verdict, expected, result.explanation)
        return result

    def test_category_and_full_log_do_not_establish_obligation(self):
        self.check('insufficient_evidence')

    def test_declined_not_required_and_pending_are_not_failures(self):
        for decision in ['declined', 'not_required', 'pending']:
            with self.subTest(decision=decision):
                self.check('not_applicable', refund={'decision': decision})

    def test_deadline_uses_snapshot_not_current_clock(self):
        refund = {'decision': 'approved', 'due_at': '2026-11-12T10:00:00Z'}
        self.check('not_applicable', refund=refund)
        self.check('fail', refund=refund, observed_at='2026-11-12T10:00:00Z')
        self.check('insufficient_evidence', refund=refund, observed_at='2026-11-12T10:00:00Z', events_complete=False)

    def test_approved_without_deadline_is_unknown(self):
        self.check('insufficient_evidence', refund={'decision': 'approved'})

    def test_claim_requires_complete_log_to_fail(self):
        refund = {'completion_claim_message_id': 'm2'}
        result = self.check('fail', refund=refund)
        self.assertEqual(result.evidence[0].ref, 'm2')
        self.check('insufficient_evidence', refund=refund, events_complete=False)

    def test_conflicting_refund_metadata_requires_review(self):
        self.check('insufficient_evidence', refund={'decision': 'declined', 'completion_claim_message_id': 'm2'})

    def test_completion_after_claim_does_not_erase_violation(self):
        self.check('fail', refund={'completion_claim_message_id': 'm2'}, events=[event(at='2026-11-12T09:11:00Z')])
        self.check('pass', refund={'completion_claim_message_id': 'm2'}, events=[event()], events_complete=False)

    def test_operation_matching_and_ambiguous_events(self):
        refund = {'completion_claim_message_id': 'm2', 'operation_id': 'refund-1'}
        self.check('fail', refund=refund, events=[event(operation_id='refund-2')])
        self.check('pass', refund=refund, events=[event(operation_id='refund-1')])
        self.check('insufficient_evidence', refund=refund, events=[event()])
        self.check('insufficient_evidence', events=[event(operation_id='refund-1')])

    def test_truthful_completion_claim_does_not_erase_missed_deadline(self):
        refund = {'decision': 'approved', 'due_at': '2026-11-12T09:05:00Z', 'completion_claim_message_id': 'm2', 'owner_id': 'owner'}
        self.check('fail', refund=refund, events=[event()])
        # The claimant is not responsible for another actor's deadline.
        self.check('pass', Scope(kind='actor', ref='agent'), refund=refund, events=[event()])

    def test_unrelated_actor_is_not_responsible(self):
        self.check('not_applicable', Scope(kind='actor', ref='another'), refund={'owner_id': 'agent', 'completion_claim_message_id': 'm2'})
        self.check('insufficient_evidence', Scope(kind='actor', ref='agent'))
        self.check('not_applicable', Scope(kind='message', ref='m2'))

    def test_owner_not_blamed_for_another_actors_claim(self):
        self.check('not_applicable', Scope(kind='actor', ref='owner'), refund={
            'decision': 'approved', 'owner_id': 'owner', 'completion_claim_message_id': 'm2', 'due_at': '2026-11-12T11:00:00Z',
        })

    def test_system_can_execute_owners_operation(self):
        self.check('pass', Scope(kind='actor', ref='agent'), refund={
            'owner_id': 'agent', 'operation_id': 'r1', 'completion_claim_message_id': 'm2',
        }, events=[event(operation_id='r1')])

    def test_legacy_positive_event_still_supported(self):
        self.check('pass', events=[event()])

    def test_input_validates_claim_and_snapshot(self):
        for changes in [
            {'refund': {'completion_claim_message_id': 'missing'}},
            {'refund': {'completion_claim_message_id': 'm1'}},
            {'observed_at': '2026-11-12T08:59:00Z'},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ticket(**changes)


class CoverageTests(unittest.TestCase):
    def test_criterion_changes_only_redistribute_its_group_weight(self):
        groups = [{'id': 'diagnosis', 'weight': 30}, {'id': 'response', 'weight': 70}]
        passed = {'group_id': 'response', 'weight': 2, 'verdict': 'pass', 'score': 1}
        failed = {'group_id': 'diagnosis', 'weight': 5, 'verdict': 'fail', 'score': 0}
        self.assertEqual(assessment_metrics([passed, failed], groups)['score'], 70)
        self.assertEqual(assessment_metrics([passed, failed, {**failed, 'weight': 50}], groups)['score'], 70)
        self.assertEqual(assessment_metrics([passed, failed, {**failed, 'verdict': 'pass', 'score': 1}], groups)['score'], 85)

    def test_group_weights_apply_to_partial_coverage_and_exclude_inapplicable_groups(self):
        groups = [{'id': 'diagnosis', 'weight': 30}, {'id': 'response', 'weight': 70}, {'id': 'controls', 'weight': 0}]
        failed = {'group_id': 'diagnosis', 'weight': 1, 'verdict': 'fail', 'score': 0}
        missing = {**failed, 'verdict': 'insufficient_evidence', 'score': None}
        passed = {'group_id': 'response', 'weight': 10, 'verdict': 'pass', 'score': 1}
        control = {**missing, 'group_id': 'controls', 'weight': 100}
        metrics = assessment_metrics([failed, missing, passed, control], groups)
        self.assertEqual(metrics['score'], 70)
        self.assertEqual(metrics['weighted_coverage'], 85)
        inapplicable = {**missing, 'verdict': 'not_applicable'}
        metrics = assessment_metrics([inapplicable, passed, control], groups)
        self.assertEqual(metrics['applicable_weight'], 70)
        self.assertEqual(metrics['weighted_coverage'], 100)
        metrics = assessment_metrics([missing, passed, control], groups)
        self.assertEqual(metrics['score'], 100)
        self.assertEqual(metrics['weighted_coverage'], 70)

    def test_model_timeout_keeps_code_checks_and_explicit_outcome(self):
        class TimeoutProvider(Provider):
            def assess(self, request):
                raise httpx.ReadTimeout('test timeout')
        result = evaluate(ticket(events=[event('issue_resolved', at='2026-11-12T10:00:00Z')]).model_dump(mode='json'),
                          {'scope': {'kind': 'conversation'}, 'card': CARD, 'knowledge': []}, TimeoutProvider())
        self.assertTrue(result['review_required'])
        self.assertEqual(result['weighted_coverage'], 0)
        self.assertEqual(result['outcome']['status'], 'resolved')
        self.assertEqual(result['timing'][0]['status'], 'met')

    def test_timing_deadlines_report_delays_without_affecting_quality(self):
        config = {
            'scope': {'kind': 'conversation'},
            'card': {**CARD, 'timing_targets': {
                'first_response': {'default_seconds': 900, 'channel_seconds': {'chat': 300}},
                'time_to_close': {'default_seconds': 1800},
            }},
            'knowledge': [],
        }
        data = ticket(created_at='2026-11-12T08:55:00Z', closed_at='2026-11-12T09:40:00Z').model_dump(mode='json')
        result = evaluate(data, config, Provider())
        timing = {row['id']: row for row in result['timing']}
        self.assertEqual((timing['first_response']['status'], timing['first_response']['actual_seconds'],
                          timing['first_response']['delay_seconds']), ('late', 600, 300))
        self.assertEqual((timing['time_to_close']['status'], timing['time_to_close']['actual_seconds'],
                          timing['time_to_close']['delay_seconds']), ('late', 2700, 900))
        self.assertEqual(result['score'], 100)
        self.assertNotIn('first_response', {row['criterion_id'] for row in result['criteria']})
        data['closed_at'] = None
        self.assertEqual(evaluate(data, config, Provider())['timing'][1]['status'], 'unavailable')

    def test_weighted_coverage_excludes_na_but_includes_missing(self):
        metrics = coverage_metrics([
            {'weight': 10, 'verdict': 'pass', 'score': 1},
            {'weight': 20, 'verdict': 'insufficient_evidence', 'score': None},
            {'weight': 70, 'verdict': 'not_applicable', 'score': None},
            {'weight': 0, 'verdict': 'fail', 'score': 0},
        ])
        self.assertEqual(metrics['weighted_coverage'], 33)
        self.assertEqual(metrics['coverage'], 67)
        self.assertEqual((metrics['checked_weight'], metrics['applicable_weight']), (10, 30))

    def test_empty_and_zero_weight_are_not_perfect_coverage(self):
        for items in [[], [{'weight': 0, 'verdict': 'pass', 'score': 1}], [{'weight': 10, 'verdict': 'not_applicable', 'score': None}]]:
            self.assertIsNone(coverage_metrics(items)['weighted_coverage'])

    def test_standard_card_has_nine_weighted_criteria_and_three_controls(self):
        self.assertEqual(sum(c['weight'] for c in CARD['criteria']), 95)
        self.assertEqual(sum(c['weight'] > 0 for c in CARD['criteria']), 9)
        self.assertEqual(len({c['id'] for c in CARD['criteria']}), 12)

    def test_no_model_does_not_invent_scores_or_outcome(self):
        result = evaluate(ticket().model_dump(mode='json'), {'scope': {'kind': 'conversation'}, 'card': CARD, 'knowledge': []})
        self.assertEqual(result['weighted_coverage'], 0)
        self.assertIsNone(result['score'])
        self.assertEqual(result['critical'], 'uncertain')
        self.assertEqual(result['outcome']['status'], 'unknown')

    def test_critical_gate_does_not_change_quality_score(self):
        data = ticket(refund={'completion_claim_message_id': 'm2'}).model_dump(mode='json')
        config = {'scope': {'kind': 'conversation'}, 'card': {'criteria': [criterion('first_response'), criterion('action_trace')]}, 'knowledge': []}
        result = evaluate(data, config)
        self.assertEqual(result['score'], 100)
        self.assertEqual(result['critical'], 'violation')
        self.assertTrue(result['review_required'])

    def test_legacy_summary_derives_coverage_without_mutating_result(self):
        result = {'score': 50, 'coverage': 50, 'critical': 'clear', 'review_required': False, 'criteria': [
            {'weight': 25, 'verdict': 'pass', 'score': 1}, {'weight': 75, 'verdict': 'insufficient_evidence', 'score': None},
        ]}
        original = copy.deepcopy(result)
        run = SimpleNamespace(id='r1', created_at=datetime.now(timezone.utc), status='complete', error=None, config={'scope': {'kind': 'conversation'}}, result=result, fingerprint='ab')
        rev = SimpleNamespace(id='v1', data={'subject': 'Test', 'category': 'general', 'channel': 'chat'})
        conv = SimpleNamespace(source='test', external_id='1')
        summary = run_summary(run, rev, conv, False)
        self.assertEqual(summary['weighted_coverage'], 25)
        self.assertIsNone(summary['outcome_status'])
        self.assertEqual(result, original)


class ModelAndScopeTests(unittest.TestCase):
    def test_single_message_cannot_see_future_messages_or_events(self):
        data = ticket().model_dump(mode='json')
        data['messages'].append({'id': 'm3', 'actor_id': 'customer', 'author': 'Customer', 'role': 'customer', 'at': '2026-11-12T09:30:00Z', 'text': 'That did not help.'})
        data['events'] = [event(at='2026-11-12T09:40:00Z')]
        value = Ticket.model_validate(data)
        scope = Scope(kind='message', ref='m2')
        provider = Provider()
        model_check(criterion('ownership'), value, scoped_messages(value, scope), [], provider, scope)
        request = provider.requests[0]
        self.assertEqual({s['ref'] for s in request['sources']}, {'m1', 'm2'})
        self.assertFalse(request['context']['messages_complete'])
        self.assertEqual(request['context']['observed_at'], '2026-11-12T09:10:00+00:00')

    def test_invalid_quote_cannot_produce_score(self):
        value = ticket()
        provider = Provider(evidence=[Evidence(type='message', ref='m2', quote='Invented quote')])
        result, _ = model_check(criterion('clarity'), value, scoped_messages(value, Scope()), [], provider)
        self.assertEqual(result.verdict, 'insufficient_evidence')

    def test_display_equivalent_quote_is_restored_from_cited_message(self):
        data = ticket().model_dump(mode='json')
        data['messages'][1]['text'] = 'Internal marker: &gt;&gt;  The refund was issued. End.'
        value = Ticket.model_validate(data)
        target = scoped_messages(value, Scope())
        provider = Provider(evidence=[Evidence(type='message', ref='m2', quote='>> The refund was issued.')])
        result, _ = model_check(criterion('clarity'), value, target, [], provider)
        self.assertEqual(result.verdict, 'pass')
        self.assertEqual(result.evidence[0].quote, '&gt;&gt;  The refund was issued.')
        for ref, quote in [('m1', '>> The refund was issued.'), ('m2', '>> The refund was accepted.')]:
            provider.evidence = [Evidence(type='message', ref=ref, quote=quote)]
            result, _ = model_check(criterion('clarity'), value, target, [], provider)
            self.assertEqual(result.verdict, 'insufficient_evidence')

    def test_scored_verdict_without_support_evidence_is_rejected(self):
        value = ticket()
        target = scoped_messages(value, Scope())
        provider = Provider(evidence=[])
        result, _ = model_check(criterion('clarity'), value, target, [], provider)
        self.assertEqual(result.verdict, 'insufficient_evidence')
        self.assertIn('evaluated support reply', result.explanation)

    def test_hybrid_still_requires_linked_knowledge(self):
        value = ticket()
        target = scoped_messages(value, Scope())
        source = {'type': 'knowledge', 'ref': 'original-ref', 'message_id': 'm2', 'text': 'Refund policy.'}
        provider = Provider(evidence=[Evidence(type='message', ref='m2', quote='The refund was issued.'), Evidence(type='knowledge', ref='k1', quote='Refund policy.')])
        result, _ = model_check(criterion('policy_accuracy'), value, target, [source], provider)
        self.assertEqual(result.verdict, 'pass')
        self.assertEqual(result.evidence[-1].ref, 'original-ref')
        result, _ = model_check(criterion('policy_accuracy'), value, target, [], provider)
        self.assertEqual(result.verdict, 'insufficient_evidence')

    def test_unknown_code_rule_does_not_fall_through_to_sla(self):
        with self.assertRaises(ValueError):
            code_check({'id': 'unknown'}, ticket(), Scope(), [])

    def test_channel_sla_and_partial_history_are_not_guessed(self):
        for changes in [{'channel': 'twitter'}, {'messages_complete': False}]:
            value = ticket(**changes)
            result = code_check(criterion('first_response'), value, Scope(), scoped_messages(value, Scope()))
            self.assertEqual(result.verdict, 'insufficient_evidence')


class OutcomeTests(unittest.TestCase):
    def test_closed_and_refund_event_do_not_prove_full_resolution(self):
        result, _ = evaluate_outcome(ticket(events=[event()]), None)
        self.assertEqual(result['status'], 'unknown')

    def test_explicit_terminal_events_work_without_model(self):
        for kind, expected in [('issue_resolved', 'resolved'), ('issue_unresolved', 'unresolved'), ('case_escalated', 'escalated')]:
            with self.subTest(kind=kind):
                result, _ = evaluate_outcome(ticket(events=[event(kind, at='2026-11-12T09:20:00Z')]), None)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['evidence'][0]['ref'], 'e1')

    def test_later_dialogue_prevents_blind_event_success(self):
        result, _ = evaluate_outcome(ticket(events=[event('issue_resolved')]), None)
        self.assertEqual(result['status'], 'unknown')

    def test_conflicting_terminal_events_are_not_guessed(self):
        result, _ = evaluate_outcome(ticket(events=[event('issue_resolved', at='2026-11-12T09:20:00Z'), event('issue_unresolved', id='e2', at='2026-11-12T09:20:00Z')]), None)
        self.assertEqual(result['status'], 'unknown')

    def test_agent_claim_cannot_prove_resolved(self):
        provider = Provider(outcome='resolved', evidence=[Evidence(type='message', ref='m2', quote='The refund was issued.')])
        result, _ = evaluate_outcome(ticket(), provider)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['review_required'])

    def test_customer_confirmation_and_all_other_statuses(self):
        data = ticket().model_dump(mode='json')
        data['messages'].append({'id': 'm3', 'actor_id': 'customer', 'author': 'Customer', 'role': 'customer', 'at': '2026-11-12T09:30:00Z', 'text': 'I received the money and the issue is resolved.'})
        provider = Provider(outcome='resolved', evidence=[Evidence(type='message', ref='m3', quote='I received the money and the issue is resolved.')])
        result, _ = evaluate_outcome(Ticket.model_validate(data), provider)
        self.assertEqual(result['status'], 'resolved')
        for status in ['solution_proposed', 'escalated', 'unresolved', 'unknown']:
            provider = Provider(outcome=status, evidence=[Evidence(type='message', ref='m2', quote='The refund was issued.')])
            result, _ = evaluate_outcome(ticket(), provider)
            self.assertEqual(result['status'], status)

    def test_outcome_requires_valid_citations(self):
        for evidence in [[], [Evidence(type='message', ref='missing', quote='missing')]]:
            result, _ = evaluate_outcome(ticket(), Provider(outcome='solution_proposed', evidence=evidence))
            self.assertEqual(result['status'], 'unknown')
            self.assertTrue(result['review_required'])

    def test_outcome_restores_display_equivalent_quote(self):
        data = ticket().model_dump(mode='json')
        data['messages'][0]['text'] = 'The issue is resolved &amp;  I received the money.'
        provider = Provider(outcome='resolved', evidence=[
            Evidence(type='message', ref='m1', quote='resolved & I received the money.'),
        ])
        result, _ = evaluate_outcome(Ticket.model_validate(data), provider)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['evidence'][0]['quote'], 'resolved &amp;  I received the money.')

    def test_outcome_review_requires_status_and_comment(self):
        ReviewInput(action='correct_outcome', outcome_status='unresolved', comment='The customer confirmed the issue.')
        for data in [dict(action='correct_outcome', outcome_status='resolved'), dict(action='confirm', outcome_status='resolved')]:
            with self.assertRaises(ValidationError):
                ReviewInput(**data)


if __name__ == '__main__':
    unittest.main()

"""Run with DATABASE_URL pointing to an isolated test database."""
import copy
import unittest
from hashlib import sha256
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from qa.api import app, access
from qa.contracts import Ticket
from qa.db import Base, Conversation, Knowledge, KnowledgeIndex, Outbox, Review, Revision, Run, Session, engine
from qa.evaluation import evaluate
from qa.scorecards import english_card
from qa.settings import CARD
from test_evaluation import LEGACY_FIRST_RESPONSE


class ScorecardAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(engine)

    def setUp(self):
        app.dependency_overrides[access] = lambda: 'Test reviewer'
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(app.dependency_overrides.clear)
        self.source = 'test-card-' + str(uuid4())
        self.addCleanup(self.cleanup_rows)
        self.data = {
            'source': self.source, 'external_id': '1', 'subject': 'Refund', 'category': 'refund',
            'messages_complete': True, 'events_complete': True,
            'refund': {'decision': 'approved', 'completion_claim_message_id': 'm2', 'owner_id': 'a'},
            'messages': [
                {'id': 'm1', 'actor_id': 'c', 'author': 'Customer', 'role': 'customer', 'at': '2026-11-12T09:00:00Z', 'text': 'Please refund my money.'},
                {'id': 'm2', 'actor_id': 'a', 'author': 'Agent', 'role': 'agent', 'at': '2026-11-12T09:01:00Z', 'text': 'The refund was issued.'},
            ],
        }

    def cleanup_rows(self):
        with Session.begin() as s:
            conversations = select(Conversation.id).where(Conversation.source == self.source)
            revisions = select(Revision.id).where(Revision.conversation_id.in_(conversations))
            runs = select(Run.id).where(Run.revision_id.in_(revisions))
            s.execute(delete(Review).where(Review.run_id.in_(runs)))
            s.execute(delete(Outbox).where(Outbox.run_id.in_(runs)))
            s.execute(delete(Run).where(Run.id.in_(runs)))
            s.execute(delete(Revision).where(Revision.id.in_(revisions)))
            s.execute(delete(Conversation).where(Conversation.id.in_(conversations)))

    def import_run(self):
        response = self.client.post('/api/import', json=self.data)
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['items'][0]['run_id']
        with Session.begin() as s:
            run = s.get(Run, run_id)
            run.result = evaluate(s.get(Revision, run.revision_id).data, run.config)
            run.status = 'complete'
        return run_id

    def test_submit_json_returns_pollable_run_and_deduplicates_import(self):
        self.data['created_at'] = '2026-11-12T08:58:00Z'
        self.data['closed_at'] = '2026-11-12T09:20:00Z'
        response = self.client.post('/api/runs', json=self.data)
        self.assertEqual(response.status_code, 202, response.text)
        item, = response.json()['items']
        self.assertEqual(response.headers['location'], item['status_url'])
        self.assertEqual(item['status'], 'queued')
        self.assertEqual(self.client.get(item['status_url']).json()['ticket']['closed_at'], self.data['closed_at'])

        second = dict(self.data, external_id='0')
        response = self.client.post('/api/runs', json=[self.data, second, self.data])
        self.assertEqual(response.status_code, 202, response.text)
        items = response.json()['items']
        self.assertEqual([i['external_id'] for i in items], ['1', '0', '1'])
        self.assertEqual(items[0], items[2])
        self.assertEqual(self.client.post('/api/import', json=self.data).json()['items'][0]['run_id'], item['run_id'])
        with Session() as session:
            tasks = list(session.scalars(select(Outbox).where(Outbox.run_id.in_([i['run_id'] for i in items]))))
            self.assertEqual(len(tasks), 2)

    def test_submit_rejects_open_or_oversized_batches(self):
        response = self.client.post('/api/runs', json=[self.data, dict(self.data, external_id='open', status='open')])
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()['detail']['external_ids'], ['open'])
        with Session() as session:
            self.assertIsNone(session.scalar(select(Conversation).where(Conversation.source == self.source)))
        self.assertEqual(self.client.post('/api/runs', json=[]).status_code, 422)
        with patch('qa.api.MAX_IMPORT', 10):
            self.assertEqual(self.client.post('/api/runs', json=self.data).status_code, 413)

    def test_knowledge_preview_matches_new_run_snapshot_and_historical_reply_dates(self):
        self.data['product'] = document = self.source
        self.addCleanup(self.cleanup_knowledge, document)
        rule = {
            'document': document, 'version': 1, 'title': 'Refund handling', 'product': document,
            'category': 'refund', 'effective_from': '2026-11-01T00:00:00Z',
            'available_from': '2026-11-01T00:00:00Z', 'fragments': {'timing': 'Refund within ten days.'},
        }
        self.assertEqual(self.client.post('/api/knowledge', json=rule).status_code, 201)
        first = self.client.post('/api/runs', json=self.data).json()['items'][0]
        self.assertEqual(self.client.post('/api/knowledge', json=rule).status_code, 409)
        future = dict(rule, version=2, available_from='2026-11-13T00:00:00Z')
        self.assertEqual(self.client.post('/api/knowledge', json=future).status_code, 201)
        library = self.client.get('/api/knowledge').json()
        self.assertEqual([r['version'] for r in library['items'] if r['document'] == document], [2, 1])
        preview = self.client.get('/api/knowledge/preview', params={'run_id': first['run_id']}).json()
        config = self.client.get(first['status_url']).json()['config']
        self.assertEqual(preview['items'], config['knowledge'])
        self.assertEqual([r['id'] for r in preview['replies']], ['m2'])
        self.assertEqual([r['version'] for r in preview['items']], [1])
        self.assertEqual(config['request_outcomes_version'], 1)
        self.assertEqual(self.client.get('/api/knowledge/preview', params={'run_id': 'missing'}).status_code, 404)

    def cleanup_knowledge(self, document):
        with Session.begin() as session:
            session.execute(delete(Knowledge).where(Knowledge.document == document))

    def test_markdown_preview_and_publish_reuse_versioned_policy_search(self):
        self.data['product'] = document = self.source
        self.addCleanup(self.cleanup_knowledge, document)
        markdown = ('# Refunds\n\n## Window\n\nRefunds take ten days.\n\n'
                    '```txt\n# A line inside an example\n```\n\n'
                    'Exceptions\n----------\n\nCard refunds exempt from this rule.\n')
        preview = self.client.post('/api/knowledge/markdown/preview', json={'markdown': markdown})
        self.assertEqual(preview.status_code, 200, preview.text)
        preview = preview.json()['fragments']
        self.assertEqual(list(preview), ['section-01', 'section-02'])
        self.assertIn('Refunds › Window', preview['section-01'])
        self.assertIn('# A line inside an example', preview['section-01'])
        self.assertNotIn(document, [row['document'] for row in self.client.get('/api/knowledge').json()['items']])
        payload = {
            'markdown': markdown, 'document': document, 'version': 1, 'title': 'Refunds',
            'product': document, 'category': 'refund', 'effective_from': '2026-11-01T00:00:00Z',
            'available_from': '2026-11-01T00:00:00Z',
        }
        self.assertEqual(self.client.post('/api/knowledge/markdown', json=payload).status_code, 201)
        self.assertEqual(self.client.post('/api/knowledge/markdown', json=payload).status_code, 409)
        rejected = self.client.post('/api/knowledge/markdown/preview', json={'markdown': '## Too long\n' + 'a' * 5000})
        self.assertEqual(rejected.status_code, 422)
        self.assertIn('Too long', rejected.json()['detail'])
        item = self.client.post('/api/runs', json=self.data).json()['items'][0]
        snapshot = self.client.get(item['status_url']).json()['config']['knowledge']
        self.assertEqual({item['text'] for item in snapshot}, set(preview.values()))

    def test_bundle_preview_routes_valid_versions_and_falls_back_to_the_library(self):
        org = self.source
        with patch('qa.api.ORG', org):
            self.run_bundle_preview_test(org)
        with Session.begin() as session:
            session.execute(delete(KnowledgeIndex).where(KnowledgeIndex.org == org))
            session.execute(delete(Knowledge).where(Knowledge.org == org))

    def run_bundle_preview_test(self, org):
        base = {'effective_from': '2026-11-01T00:00:00Z', 'available_from': '2026-11-01T00:00:00Z',
                'language': 'en', 'version': 1}
        rule = {**base, 'document': org + '-expiry', 'title': 'Password reset links', 'path': 'account/reset.md',
                'markdown': '## Expiration\nA password reset link is valid for 45 minutes.'}
        other = {**base, 'document': org + '-other', 'title': 'Shipping', 'path': 'shipping.md',
                 'markdown': '## Delivery fees\nExpress delivery fees are paid before dispatch.'}
        bad_index = '# Access\n\n[Password reset](../account/reset.md)'
        self.assertEqual(self.client.post('/api/knowledge/bundle', json={'index': bad_index, 'documents': [rule]}).status_code, 422)

        initial = {'index': '# Password reset\n\n[Link expiry](account/reset.md)\n\n# Orders\n\n[Shipping](shipping.md)',
                   'documents': [rule, other]}
        preview = self.client.post('/api/knowledge/bundle/preview', json=initial)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(len(preview.json()['sections']), 2)
        self.assertEqual(self.client.get('/api/knowledge').json()['items'], [])
        published = self.client.post('/api/knowledge/bundle', json=initial)
        self.assertEqual(published.status_code, 201, published.text)
        self.assertEqual(self.client.post('/api/knowledge/bundle', json=initial).status_code, 201)

        self.data.update(language='en', subject='Password reset link expiry')
        self.data['messages'][0]['text'] = 'How long is the reset link valid?'
        self.data['messages'][1]['text'] = 'The password reset link is valid for 45 minutes.'
        submitted = self.client.post('/api/runs', json=self.data).json()['items'][0]
        snapshot = self.client.get(submitted['status_url']).json()['config']['knowledge']
        self.assertEqual([item['search_mode'] for item in snapshot], ['section'])
        self.assertIn('45 minutes', snapshot[0]['text'])
        fallback = {**initial, 'index': '# Password reset\n\n[Shipping](shipping.md)\n\n# Archive\n\n[Old forms](account/reset.md)'}
        self.assertEqual(self.client.post('/api/knowledge/bundle', json=fallback).status_code, 201)
        matched = self.client.get('/api/knowledge/preview', params={'run_id': submitted['run_id']}).json()['items']
        self.assertEqual([item['search_mode'] for item in matched], ['fallback'])
        self.assertEqual(matched[0]['ref'], snapshot[0]['ref'])
        self.assertEqual(self.client.get(submitted['status_url']).json()['config']['knowledge'], snapshot)

        conflicting = {**fallback, 'documents': [dict(rule, markdown='This differs from the stored version.'), other]}
        self.assertEqual(self.client.post('/api/knowledge/bundle', json=conflicting).status_code, 409)
        self.assertEqual(self.client.get('/api/knowledge/preview', params={'run_id': submitted['run_id']}).json()['items'], matched)

    def test_legacy_scorecard_translation_preserves_customizations_and_history(self):
        historical = copy.deepcopy(CARD)
        historical['name'] = 'Earlier scorecard name'
        historical['groups'][0]['name'] = 'Earlier group name'
        historical['criteria'][0]['description'] = 'Earlier criterion description'
        def fingerprint(value):
            return sha256(value.encode()).hexdigest()
        fingerprints = {
            'name': fingerprint(historical['name']),
            'groups': {historical['groups'][0]['id']: {'name': fingerprint(historical['groups'][0]['name'])}},
            'criteria': {historical['criteria'][0]['id']: {
                'description': fingerprint(historical['criteria'][0]['description']),
            }},
        }
        migration = patch('qa.scorecards.LEGACY_CARD_FINGERPRINTS', fingerprints)
        migration.start()
        self.addCleanup(migration.stop)
        historical["groups"][0]["weight"] = 42
        historical["criteria"][0]["name"] = "Custom criterion"
        historical["criteria"][0]["rubric"]["pass"] = "Custom pass rule"
        localized = english_card(historical)

        self.assertEqual(localized["name"], CARD["name"])
        self.assertEqual(localized["version"], historical["version"])
        self.assertEqual(localized["groups"][0], {**CARD["groups"][0], "weight": 42})
        self.assertEqual(localized["criteria"][0]["name"], "Custom criterion")
        self.assertEqual(localized["criteria"][0]["rubric"]["pass"], "Custom pass rule")
        self.assertEqual(localized["criteria"][0]["description"], CARD["criteria"][0]["description"])
        self.assertEqual(historical["name"], 'Earlier scorecard name')

        run_id = self.import_run()
        with Session.begin() as s:
            run = s.get(Run, run_id)
            run.config = {**run.config, "card": historical}
        data = self.client.get(f"/api/runs/{run_id}").json()
        self.assertEqual(data["config"]["card"], historical)
        self.assertEqual(data["display_names"]["groups"]["solution"], CARD["groups"][0]["name"])
        self.assertEqual(data["display_names"]["criteria"]["request_understanding"], "Custom criterion")

    def test_import_assessment_review_export_and_rerun(self):
        run_id = self.import_run()
        response = self.client.get('/api/runs/' + run_id)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        original = copy.deepcopy(data['result'])
        self.assertEqual(data['config']['card']['version'], CARD['version'])
        self.assertEqual(data['weighted_coverage'], 0)
        self.assertEqual(data['outcome_status'], 'unknown')
        self.assertEqual(data['critical'], 'violation')
        self.assertEqual(data['ticket']['refund']['owner_id'], 'a')
        decision = {'action': 'correct_outcome', 'outcome_status': 'unresolved', 'comment': 'The customer confirmed they did not receive the money.'}
        response = self.client.post(f'/api/runs/{run_id}/reviews', json=decision)
        self.assertEqual(response.status_code, 201, response.text)
        exported = self.client.get('/api/runs/' + run_id).json()
        self.assertEqual(exported['result'], original)
        self.assertEqual(exported['reviews'][0]['outcome_status'], 'unresolved')
        self.assertTrue(exported['reviewed'])
        response = self.client.post(f'/api/runs/{run_id}/rerun', json={'scope': {'kind': 'message', 'ref': 'm2'}})
        self.assertEqual(response.status_code, 202, response.text)
        new_run = self.client.get('/api/runs/' + response.json()['id']).json()
        self.assertEqual(new_run['config']['card']['version'], CARD['version'])
        self.assertEqual(new_run['scope'], {'kind': 'message', 'ref': 'm2'})

    def test_invalid_refund_metadata_rejected_at_import(self):
        self.data['refund']['completion_claim_message_id'] = 'missing'
        self.assertEqual(self.client.post('/api/import', json=self.data).status_code, 422)

    def test_review_legacy_run_requires_new_assessment(self):
        run_id = self.import_run()
        with Session.begin() as s:
            run = s.get(Run, run_id)
            result = copy.deepcopy(run.result)
            for key in ['outcome', 'weighted_coverage', 'checked_weight', 'applicable_weight']:
                result.pop(key)
            run.result = result
        response = self.client.get('/api/runs/' + run_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['weighted_coverage'], 0)
        self.assertIsNone(response.json()['outcome_status'])
        response = self.client.post(f'/api/runs/{run_id}/reviews', json={
            'action': 'correct_outcome', 'outcome_status': 'resolved', 'comment': 'Checked.',
        })
        self.assertEqual(response.status_code, 422)

    def test_published_group_weights_reach_the_evaluator_and_run_response(self):
        from qa.db import Scorecard
        from qa.settings import ORG

        card = copy.deepcopy(CARD)
        card['version'] += 1
        card['groups'] = [{'id': 'speed', 'name': 'Speed', 'weight': 30},
                          {'id': 'reply', 'name': 'Reply', 'weight': 70}]
        criteria = {item['id']: item for item in card['criteria']}
        card['criteria'] = [
            {**criteria['action_trace'], 'group_id': 'speed', 'weight': 100},
            {**criteria['response_present'], 'group_id': 'reply', 'weight': 1},
        ]
        response = self.client.put('/api/card', json={
            'base_version': CARD['version'],
            **{key: card[key] for key in ('name', 'groups', 'criteria')},
        })
        self.assertEqual(response.status_code, 200, response.text)
        try:
            self.data['messages'][1]['at'] = '2026-11-12T09:30:00Z'
            run_id = self.import_run()
            run = self.client.get('/api/runs/' + run_id).json()
            self.assertEqual(run['config']['card']['groups'], card['groups'])
            self.assertEqual(run['score'], 70)
            self.assertEqual(run['weighted_coverage'], 100)
        finally:
            with Session.begin() as session:
                session.execute(delete(Scorecard).where(Scorecard.org == ORG, Scorecard.version == card['version']))


if __name__ == '__main__':
    unittest.main()

class ConfigurableCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(engine)

    def setUp(self):
        from qa.db import Scorecard
        from qa.settings import ORG
        self.Scorecard, self.org = Scorecard, ORG
        app.dependency_overrides[access] = lambda: 'Test editor'
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(app.dependency_overrides.clear)
        self.addCleanup(self.cleanup_cards)
        self.card = self.client.get('/api/settings').json()['card']
        self.payload = {key: copy.deepcopy(self.card[key]) for key in ['name', 'groups', 'criteria']}
        self.payload['base_version'] = self.card['version']

    def cleanup_cards(self):
        with Session.begin() as s:
            s.execute(delete(self.Scorecard).where(self.Scorecard.org == self.org, self.Scorecard.version > self.card['version']))

    def test_group_custom_criterion_and_weights_persist(self):
        self.payload['groups'].append({'id': 'custom_group', 'name': 'Our requirements', 'weight': 20})
        custom = {
            'id': 'custom_empathy', 'group_id': 'custom_group', 'name': 'Accounts for the context',
            'description': 'Accounts for the urgency of the request.', 'applicability': 'The customer says the request is urgent.',
            'weight': 7.5, 'rubric': {'pass': 'The reply accounts for urgency.', 'fail': 'The reply ignores explicit urgency.'},
        }
        self.payload['criteria'].append(custom)
        response = self.client.put('/api/card', json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()
        self.assertEqual(saved['version'], self.card['version'] + 1)
        self.assertEqual(saved['groups'][-1]['name'], 'Our requirements')
        self.assertEqual(saved['groups'][-1]['weight'], 20)
        self.assertEqual(saved['criteria'][-1]['weight'], 7.5)
        self.assertEqual(saved['criteria'][-1]['evaluator_type'], 'llm')
        self.assertEqual(self.client.get('/api/settings').json()['card'], saved)
        # A stale editor cannot overwrite the new version.
        self.assertEqual(self.client.put('/api/card', json=self.payload).status_code, 409)
        self.payload['base_version'] = saved['version']
        self.payload['criteria'][-1]['weight'] = 15
        second = self.client.put('/api/card', json=self.payload).json()
        self.assertEqual(second['groups'][-1]['weight'], 20)
        self.assertEqual(second['criteria'][-1]['version'], 2)
        with Session() as s:
            previous = s.get(self.Scorecard, (self.org, saved['version']))
            self.assertEqual(previous.data['criteria'][-1]['weight'], 7.5)

    def test_invalid_cards_cannot_be_published(self):
        for modify in [
            lambda p: p['groups'][0].update(weight=-1),
            lambda p: [g.update(weight=0) for g in p['groups']],
            lambda p: p['groups'].append({'id': 'empty_group', 'name': 'Empty group', 'weight': 10}),
            lambda p: p['criteria'][0].update(weight=-1),
            lambda p: p['criteria'][0].update(group_id='missing'),
            lambda p: p['criteria'].append(copy.deepcopy(p['criteria'][0])),
            lambda p: p['criteria'][0].update(rubric={'pass': 'Yes'}),
            lambda p: [c.update(weight=0) for c in p['criteria']],
            lambda p: p['criteria'][0].update(evaluator_type='code'),
        ]:
            payload = copy.deepcopy(self.payload)
            modify(payload)
            response = self.client.put('/api/card', json=payload)
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.get('/api/settings').json()['card']['version'], self.card['version'])

    def test_first_save_uses_previous_totals_for_legacy_groups(self):
        old_card = copy.deepcopy(self.card)
        old_card['version'] += 1
        for group in old_card['groups']:
            group.pop('weight')
        with Session.begin() as session:
            session.add(self.Scorecard(org=self.org, version=old_card['version'], data=old_card))
        payload = {key: copy.deepcopy(old_card[key]) for key in ['name', 'groups', 'criteria']}
        payload['base_version'] = old_card['version']
        payload['criteria'][0]['weight'] += 5
        response = self.client.put('/api/card', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['groups'], [
            {**group, 'weight': sum(c['weight'] for c in old_card['criteria'] if c['group_id'] == group['id'])}
            for group in old_card['groups']
        ])

    def test_delete_group_and_clear_an_implicitly_weighted_group(self):
        old_card = copy.deepcopy(self.card)
        old_card['version'] += 1
        for group in old_card['groups']:
            group.pop('weight', None)
        deleted, cleared = (group['id'] for group in old_card['groups'][:2])
        with Session.begin() as session:
            session.add(self.Scorecard(org=self.org, version=old_card['version'], data=old_card))
        payload = {key: old_card[key] for key in ('name', 'timing_targets')}
        payload['base_version'] = old_card['version']
        payload['groups'] = [group for group in old_card['groups'] if group['id'] != deleted]
        payload['criteria'] = [c for c in old_card['criteria'] if c['group_id'] not in {deleted, cleared}]

        response = self.client.put('/api/card', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()
        self.assertEqual(saved['groups'], payload['groups'])
        self.assertEqual(saved['criteria'], payload['criteria'])
        with Session() as session:
            self.assertEqual(session.get(self.Scorecard, (self.org, old_card['version'])).data, old_card)

    def test_timing_targets_migrate_from_a_saved_card_and_can_be_changed(self):
        historical = copy.deepcopy(CARD)
        historical.pop('timing_targets')
        historical['criteria'].append(copy.deepcopy(LEGACY_FIRST_RESPONSE))
        historical['version'] = self.card['version'] + 1
        for group in historical['groups']:
            group.pop('weight')
        next(c for c in historical['criteria'] if c['id'] == 'first_response')['channel_threshold_seconds'] = {'chat': 120}
        with Session.begin() as session:
            session.add(self.Scorecard(org=self.org, version=historical['version'], data=historical))
        current = self.client.get('/api/settings').json()['card']
        self.assertNotIn('first_response', [c['id'] for c in current['criteria']])
        self.assertEqual(current['timing_targets']['first_response']['channel_seconds'], {'chat': 120})
        body = {k: current[k] for k in ('name', 'groups', 'criteria', 'timing_targets')}
        body['base_version'] = current['version']
        body['timing_targets']['time_to_close'] = {'default_seconds': 1800}
        response = self.client.put('/api/card', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['timing_targets']['time_to_close']['default_seconds'], 1800)
        self.assertTrue(all('weight' not in group for group in response.json()['groups']))

    def test_new_run_uses_published_card_and_existing_snapshot_is_immutable(self):
        from qa.contracts import Scope
        from qa.service import ingest, enqueue
        from qa.settings import ORG
        source = 'card-version-' + str(uuid4())
        value = Ticket(external_id='1', source=source, subject='Hello', messages=[{
            'id':'m1','actor_id':'c','author':'Customer','role':'customer','at':'2026-11-12T09:00:00Z','text':'Question',
        }])
        try:
            with Session.begin() as s:
                revision = ingest(s, ORG, value)
                old_id = enqueue(s, ORG, revision, Scope())
            self.payload['criteria'][0]['weight'] = 25
            published = self.client.put('/api/card', json=self.payload).json()
            with Session.begin() as s:
                old_run = s.get(Run, old_id)
                new_id = enqueue(s, ORG, s.get(Revision, old_run.revision_id), Scope(), force=True)
                self.assertEqual(old_run.config['card']['version'], self.card['version'])
                self.assertEqual(s.get(Run, new_id).config['card'], published)
        finally:
            with Session.begin() as s:
                conversations = select(Conversation.id).where(Conversation.source == source)
                revisions = select(Revision.id).where(Revision.conversation_id.in_(conversations))
                runs = select(Run.id).where(Run.revision_id.in_(revisions))
                s.execute(delete(Outbox).where(Outbox.run_id.in_(runs)))
                s.execute(delete(Run).where(Run.id.in_(runs)))
                s.execute(delete(Revision).where(Revision.id.in_(revisions)))
                s.execute(delete(Conversation).where(Conversation.id.in_(conversations)))

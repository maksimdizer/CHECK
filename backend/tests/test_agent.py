import unittest
from collections import Counter

from qa.agent import FRAGMENT_LENGTH, REVIEW_PROFILES, AgentProvider, CaseMap, CompletenessDecision, CriterionDecision, CustomerRequests, Read
from qa.evaluation import model_check, scoped_messages
from qa.contracts import Scope
from qa.settings import AGENT_VERSION, CARD
from test_evaluation import criterion, ticket


class ScriptedAgent(AgentProvider):
    def __init__(self, replies, *, quality=False, agent_version=8):
        super().__init__("test", num_predict=1024, prompt_version=5, quality=quality, agent_version=agent_version)
        self.replies = iter(replies)
        self.calls = []
        self.requests = []
        self.prompts = []

    def _generate(self, request, contract, prompt, *, schema=None, think=None):
        self.calls.append((contract, think))
        self.requests.append(request)
        self.prompts.append(prompt)
        self.schema = schema
        return contract.model_validate(next(self.replies)), {"eval_count": 1}


class AgentTests(unittest.TestCase):
    def test_completeness_audits_pass_for_omitted_compensation(self):
        value, scope = ticket(), Scope()
        value.messages[0].text = "Please refund my payment. Am I entitled to compensation for the outage?"
        requests = [{"question": "Refund the payment", "customer_ids": ["e1"]}]
        refund = {"question_id": 1, "coverage": "answered", "support_ids": ["e2"],
                  "basis": "The refund was issued."}
        first = {"verdict": "pass", "coverage": [refund], "evidence_ids": ["e2"], "explanation": "The refund was issued."}
        compensation = {"question": "Compensation for the outage", "customer_ids": ["e1"]}
        open_question = {"question_id": 2, "coverage": "open", "support_ids": [],
                         "basis": "The reply did not mention compensation."}
        provider = ScriptedAgent([
            {"facts": []}, {"questions": requests}, first, {"questions": [compensation]},
            {**first, "coverage": [refund, open_question]},
        ])
        answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(answer.verdict, "partial")
        self.assertIn("Compensation for the outage", answer.explanation)
        self.assertEqual(provider.requests[3]["known_questions"][0]["question"], "Refund the payment")
        self.assertEqual(provider.requests[4]["questions"][1]["question"], "Compensation for the outage")

    def test_completeness_rechecks_empty_extraction_with_support_context(self):
        value, scope = ticket(), Scope()
        value.messages[0].text = "I still have not received a refund for the defective item."
        value.messages[1].text = "Please send your order number so we can check your refund."
        request = {"question": "Refund the defective item", "customer_ids": ["e1"]}
        row = {"question_id": 1, "coverage": "next_step", "support_ids": ["e2"],
               "basis": "The order number is needed to check the refund."}
        provider = ScriptedAgent([
            {"facts": []}, {"questions": []}, {"questions": [request]},
            {"verdict": "pass", "coverage": [row], "evidence_ids": ["e2"], "explanation": "Support requested an order number."},
            {"questions": []},
        ])
        answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(answer.verdict, "pass")
        self.assertEqual(provider.calls[2], (CustomerRequests, True))
        self.assertEqual([src["id"] for src in provider.requests[2]["sources"]], ["e1", "e2"])

    def test_completeness_rejects_early_reply_without_discarding_other_questions(self):
        sources = [
            {"type": "message", "ref": "m1", "role": "customer", "text": "Please refund my payment and explain the charge."},
            {"type": "message", "ref": "m2", "role": "agent", "text": "The refund was issued."},
            {"type": "message", "ref": "m3", "role": "customer", "text": "When will the money arrive?"},
            {"type": "message", "ref": "m4", "role": "agent", "text": "Thank you for contacting us."},
        ]
        provider = ScriptedAgent([
            {"questions": [
                {"question": "Refund the payment", "customer_ids": ["e1"]},
                {"question": "Give the expected timing", "customer_ids": ["e3"]},
                {"question": "Explain the charge", "customer_ids": ["e1"]},
            ]},
            {"verdict": "partial", "evidence_ids": [], "explanation": "The requests were checked.",
             "coverage": [
                 {"question_id": 1, "coverage": "answered", "support_ids": ["e2"], "basis": "The refund was issued."},
                 {"question_id": 2, "coverage": "answered", "support_ids": ["e2"], "basis": "The refund was issued."},
                 {"question_id": 3, "coverage": "open", "support_ids": ["e4"], "basis": "The reply does not give a reason."},
             ]},
        ])
        view = provider._fragments(sources)
        request = {"sources": sources, "target_messages": ["m2", "m4"], "criterion": criterion("completeness")}
        payload = {"sources": [{"id": fid, "text": t} for fid, (_, t) in view.items()]}
        trace = {}
        answer = provider._check_completeness(payload, view, request, trace, Counter())
        self.assertEqual(answer.verdict, "partial")
        self.assertEqual(provider.requests[1]["questions"][1]["eligible_support_ids"], ["e4"])
        self.assertEqual(trace["completeness_chronology_rejected"], [2])
        self.assertEqual(trace["completeness_validated_coverage"][1]["coverage"], "unknown")
        self.assertIn("Give the expected timing", answer.explanation)

    def test_completeness_requires_coverage_for_each_question_in_same_message(self):
        value, scope = ticket(), Scope()
        value.messages[0].text = "Please refund my payment. When will the money arrive?"
        value.messages[1].text = "The refund was issued. The bank will confirm when it will be credited."
        questions = [
            {"question": "Refund the payment", "customer_ids": ["e1"]},
            {"question": "Give the expected timing", "customer_ids": ["e1"]},
        ]
        refund = {"question_id": 1, "coverage": "answered", "support_ids": ["e2"], "basis": "The refund is confirmed."}
        timing = {"question_id": 2, "coverage": "open", "support_ids": [], "basis": "The timing was not provided."}
        for rows, expected in (
            ([refund, timing], "partial"),
            ([refund, {**timing, "coverage": "unknown"}], "insufficient_evidence"),
            ([refund, {**timing, "coverage": "next_step"}], "insufficient_evidence"),
            ([refund, {**timing, "coverage": "next_step", "support_ids": ["e2"],
                       "basis": "The bank will confirm when it will be credited."}], "pass"),
            ([refund, {**timing, "coverage": "next_step", "support_ids": ["e1"]}], "insufficient_evidence"),
            ([refund], "insufficient_evidence"),
            ([refund, refund], "insufficient_evidence"),
        ):
            with self.subTest(rows=rows, expected=expected):
                review = {"verdict": "pass", "coverage": rows, "evidence_ids": [], "explanation": "The questions were checked."}
                incomplete = len({row["question_id"] for row in rows}) != len(questions)
                provider = ScriptedAgent([{"facts": []}, {"questions": questions}, review, review if incomplete else {"questions": []}])
                answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
                self.assertEqual(answer.verdict, expected)
                expected_calls = [(CaseMap, False), (CustomerRequests, False), (CompletenessDecision, True)]
                if incomplete:
                    expected_calls.append((CompletenessDecision, True))
                if expected == "pass":
                    expected_calls.append((CustomerRequests, True))
                self.assertEqual(provider.calls, expected_calls)
                self.assertEqual([row["id"] for row in provider.requests[1]["sources"]], ["e1"])
                self.assertEqual([row["question_id"] for row in provider.requests[2]["questions"]], [1, 2])
                if expected == "partial":
                    self.assertIn("Give the expected timing", answer.explanation)
                    self.assertEqual(answer.evidence[0].quote, value.messages[1].text)

    def test_completeness_requires_all_message_fragments_for_pass(self):
        first_part = "Please refund my payment. " + "A" * (FRAGMENT_LENGTH - len("Please refund my payment. "))
        sources = [
            {"type": "message", "ref": "m1", "role": "customer", "text": first_part + "And why was I charged?"},
            {"type": "message", "ref": "m2", "role": "agent", "text": "The refund was issued."},
        ]
        provider = ScriptedAgent([
            {"questions": [{"question": "Refund the payment", "customer_ids": ["e1"]}]},
            {"verdict": "pass", "evidence_ids": ["e3"], "explanation": "The refund was issued.",
             "coverage": [{"question_id": 1, "support_ids": ["e3"], "coverage": "answered", "basis": "The refund is confirmed."}]},
        ])
        view = provider._fragments(sources)
        del view["e2"]
        request = {"sources": sources, "target_messages": ["m2"], "criterion": criterion("completeness")}
        payload = {"sources": [{"id": fid, "text": text} for fid, (_, text) in view.items()]}
        answer = provider._check_completeness(payload, view, request, {}, Counter())
        self.assertEqual(answer.verdict, "insufficient_evidence")

    def test_read_tools_match_identifiers_case_sensitively(self):
        sources = [
            {"type":"message","ref":"m1","text":"My booking is P1278F."},
            {"type":"message","ref":"m2","text":"Correct number PI278F."},
        ]
        fragments = AgentProvider._fragments(sources)
        request = Read(tool="find_exact", args=["PI278F"])
        self.assertEqual(AgentProvider._read(request,sources,fragments), ["e2"])
        request = Read(tool="get_neighbors",args=["m2"])
        self.assertEqual(AgentProvider._read(request,sources,fragments), ["e1","e2"])

    def test_reuses_fact_map_and_extracts_exact_quote(self):
        provider = ScriptedAgent([
            {"facts":[{"text":"The customer requested a refund.","refs":["m1"]}]},
            {"verdict":"pass","evidence_ids":["e2"],"explanation":"The refund was issued."},
            {"verdict":"pass","evidence_ids":["e2"],"explanation":"The refund was issued."},
        ])
        value, scope = ticket(), Scope()
        target = scoped_messages(value,scope)
        for cid in ("clarity","respect"):
            answer,_ = model_check(criterion(cid),value,target,[],provider,scope)
            self.assertEqual(answer.evidence[0].quote, "The refund was issued.")
        self.assertEqual([c for c,_ in provider.calls], [CaseMap,CriterionDecision,CriterionDecision])

    def test_unresolved_disagreement_goes_to_review_without_score(self):
        provider = ScriptedAgent([
            {"facts":[]},
            {"verdict":"fail","evidence_ids":["e2"],"explanation":"The reply violates the rubric."},
            {"verdict":"insufficient_evidence","evidence_ids":[],"explanation":"The verdict is uncertain.","needs_review":True},
        ])
        value, scope = ticket(), Scope()
        answer,_ = model_check(criterion("clarity"),value,scoped_messages(value,scope),[],provider,scope)
        self.assertEqual(answer.verdict,"insufficient_evidence")
        self.assertIn("clarity",provider.manual_review)
        self.assertEqual([think for _,think in provider.calls],[False,False,True])

    def test_grouped_draft_calls_model_once_and_limits_evidence_ids(self):
        decision = {"verdict":"pass","evidence_ids":["e2"],"explanation":"The reply is clear."}
        provider = ScriptedAgent([{"facts":[]}, {"clarity":decision, "respect":decision}], agent_version=AGENT_VERSION)
        value, scope = ticket(), Scope()
        targets = scoped_messages(value, scope)
        peers = [criterion("clarity"), criterion("respect")]
        for c in peers:
            answer,_ = model_check(c,value,targets,[],provider,scope,peers=peers)
            self.assertEqual(answer.verdict,"pass")
        self.assertEqual([think for _,think in provider.calls],[False,True])
        evidence = provider.schema["$defs"]["CriterionDecision"]["properties"]["evidence_ids"]
        self.assertEqual(evidence["items"]["enum"],["e1","e2"])
        task = provider.requests[1]["task"]
        profile = REVIEW_PROFILES[AGENT_VERSION]
        self.assertEqual(task["review_sequence"], profile["sequence"])
        self.assertNotIn("checks", task)
        for peer in peers:
            self.assertEqual(task["criteria"][peer["id"]]["checks"], profile["criteria"][peer["id"]])
            self.assertEqual(task["criteria"][peer["id"]]["rubric"], peer["rubric"])

    def test_internal_checks_cover_builtins_and_custom_text_remains_authoritative(self):
        profile = REVIEW_PROFILES[AGENT_VERSION]
        model_ids = {item["id"] for item in CARD["criteria"] if item["evaluator_type"] != "code"}
        self.assertEqual(model_ids, set(profile["criteria"]))
        self.assertTrue(all(profile["criteria"].values()))
        value, scope = ticket(), Scope()
        targets = scoped_messages(value, scope)
        decision = {"verdict": "pass", "evidence_ids": ["e2"], "explanation": "The reply states the status."}
        rubric = {"pass": "The reply states the status.", "fail": "The reply omits the status."}
        for cid in ("custom_status", "completeness"):
            with self.subTest(criterion=cid):
                provider = ScriptedAgent([{"facts": []}, decision], agent_version=AGENT_VERSION)
                edited = {**criterion("completeness"), "id": cid, "rubric": rubric}
                answer, _ = model_check(edited, value, targets, [], provider, scope)
                self.assertEqual(answer.verdict, "pass")
                self.assertEqual([contract for contract, _ in provider.calls], [CaseMap, CriterionDecision])
                task = provider.requests[1]["task"]
                self.assertEqual(task["criterion"]["rubric"], rubric)
                self.assertEqual(task["checks"], profile["criteria"].get(cid, profile["fallback"]))

    def test_existing_agent_version_keeps_the_previous_prompt(self):
        value, scope = ticket(), Scope()
        provider = ScriptedAgent([{"facts": []}, {
            "verdict": "pass", "evidence_ids": ["e2"], "explanation": "The reply was clear.",
        }], agent_version=6)
        model_check(criterion("clarity"), value, scoped_messages(value, scope), [], provider, scope)
        payload = provider.requests[1]
        self.assertIn("criterion_guidance", payload)
        self.assertNotIn("review_sequence", payload["task"])
        self.assertNotIn("checks", payload["task"])

    def test_v8_explains_answer_vs_next_step_on_completeness_retry(self):
        value, scope = ticket(), Scope()
        requests = {"questions": [{"question": "When will the refund arrive?", "customer_ids": ["e1"]}]}
        result = {"verdict": "pass", "evidence_ids": ["e2"], "explanation": "Support will check.",
                  "coverage": [{"question_id": 1, "coverage": "next_step", "support_ids": ["e2"],
                                "basis": "Support will check the date."}]}
        for version in (7, 8):
            with self.subTest(agent_version=version):
                provider = ScriptedAgent([{"facts": []}, requests, {**result, "coverage": []}, result,
                                          {"questions": []}], agent_version=version)
                answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
                self.assertEqual(answer.verdict, "pass")
                for prompt in provider.prompts[2:4]:
                    self.assertEqual("Apply the coverage labels literally" in prompt, version == 8)

    def test_v9_ownership_verifies_business_days_and_material_missing_timing(self):
        scope = Scope()
        for observed_at, clock, timing, expected in (
            ("2026-11-18T15:00:00Z", {"amount": 3, "unit": "business_days", "quote": "three business days"},
             "stated", "insufficient_evidence"),
            ("2026-11-19T17:00:00Z", {"amount": 3, "unit": "business_days", "quote": "three business days"},
             "stated", "fail"),
            ("2026-11-18T15:00:00Z", None, "missing_material", "partial"),
        ):
            with self.subTest(timing=timing, observed_at=observed_at):
                value = ticket(observed_at=observed_at, messages=[
                    {"id": "m1", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": "2026-11-13T15:30:00Z", "text": "When will you call me?"},
                    {"id": "m2", "actor_id": "a", "role": "agent", "author": "Support",
                     "at": "2026-11-13T16:00:00Z",
                     "text": "Our team will call within three business days." if clock else "Our team will call shortly."},
                    {"id": "m3", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": observed_at, "text": "I still haven't been called."},
                ])
                provider = ScriptedAgent([{"facts": []}, {
                    "actions": [{"reply_id": "e2", "action": "Call the customer", "status": "missed" if clock else "pending",
                                 "timing": timing, "owner_clear": True, "conditional": False,
                                 "trigger_id": None, "deadline": clock, "status_ids": ["e3"]}],
                    "explanation": "The team has not called yet.",
                }], agent_version=9)
                answer, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [], provider, scope)
                self.assertEqual(answer.verdict, expected)
                self.assertEqual(answer.evidence[0].ref, "m2")
                if clock and expected == "insufficient_evidence":
                    provider = ScriptedAgent([{"facts": []}, {
                        "actions": [{"reply_id": "e2", "action": "Call the customer", "status": "missed",
                                     "timing": "missing_material", "owner_clear": True, "conditional": False,
                                     "trigger_id": None, "deadline": clock, "status_ids": ["e3"]}],
                        "explanation": "The deadline was missed.", "applicable": True,
                    }], agent_version=10)
                    answer, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [], provider, scope)
                    self.assertEqual(answer.verdict, "insufficient_evidence")
                    self.assertIn("does not establish", answer.explanation)

    def test_v9_records_a_concrete_followup_as_next_step(self):
        value, scope = ticket(), Scope()
        value.messages[0].text = "When will the refund arrive?"
        value.messages[1].text = "Our payments team will check the settlement date."
        provider = ScriptedAgent([
            {"facts": []},
            {"questions": [{"question": "When will the refund arrive?", "customer_ids": ["e1"]}]},
            {"verdict": "pass", "evidence_ids": ["e2"], "explanation": "The payments team will check the date.",
             "coverage": [{"question_id": 1, "support_ids": ["e2"], "basis": "Payments will check the date.",
                           "relation": "specific_next_step"}]},
            {"questions": []},
        ], agent_version=9)
        answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(answer.verdict, "pass")
        row = provider.trace["checks"]["completeness"]["completeness_review"]["coverage"][0]
        self.assertEqual(row["coverage"], "next_step")

    def test_v10_rejects_a_fail_when_every_request_has_a_valid_step(self):
        value, scope = ticket(), Scope()
        value.messages[0].text = "When will the refund arrive?"
        value.messages[1].text = "Our payments team will check the settlement date."
        provider = ScriptedAgent([
            {"facts": []},
            {"questions": [{"question": "When will the refund arrive?", "customer_ids": ["e1"]}]},
            {"verdict": "fail", "evidence_ids": ["e2"], "explanation": "The exact date was not given.",
             "coverage": [{"question_id": 1, "support_ids": ["e2"], "basis": "Payments will check the date.",
                           "relation": "specific_next_step"}]},
            {"questions": []},
        ], agent_version=10)
        answer, _ = model_check(criterion("completeness"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(answer.verdict, "pass")
        self.assertIn("specific next step", answer.explanation)

    def test_v11_rechecks_a_claimed_miss_only_against_later_evidence(self):
        value, scope = ticket(messages=[
            {"id": "m1", "actor_id": "c", "role": "customer", "author": "Customer",
             "at": "2026-11-13T15:00:00Z", "text": "When will you call?"},
            {"id": "m2", "actor_id": "a", "role": "agent", "author": "Support",
             "at": "2026-11-13T16:00:00Z", "text": "Our team will call within three business days."},
            {"id": "m3", "actor_id": "c", "role": "customer", "author": "Customer",
             "at": "2026-11-18T15:00:00Z", "text": "I still have not been called."},
            {"id": "m4", "actor_id": "c", "role": "customer", "author": "Customer",
             "at": "2026-11-19T16:00:00Z", "text": "There was no call from your team."},
        ]), Scope()
        assessment = {"applicable": True, "explanation": "The customer reports no callback.",
                      "actions": [{"reply_id": "e2", "status_ids": ["e2"], "action": "Call the customer",
                                   "conditional": False, "trigger_id": None, "status": "missed",
                                   "timing": "stated", "owner_clear": True,
                                   "deadline": {"amount": 3, "unit": "business_days", "quote": "three business days"}}]}
        provider = ScriptedAgent([{"facts": []}, assessment, {"evidence": [{"reply_id": "e2", "later_id": "e4"}]}],
                                 agent_version=11)
        result, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(provider.requests[2]["promises"]["e2"]["later_ids"], ["e4"])
        self.assertEqual(result.verdict, "fail")
        self.assertEqual([ev.ref for ev in result.evidence], ["m2", "m4"])

    def test_v11_does_not_treat_a_proposed_handoff_as_proof_it_was_blocked(self):
        value, scope = ticket(), Scope()
        value.messages[1].text = "Our team can check. Please send your order number."
        assessment = {"applicable": True, "explanation": "The handoff may be pending.",
                      "actions": [{"reply_id": "e2", "status_ids": ["e2"], "action": "Check the order",
                                   "conditional": True, "trigger_id": None, "status": "blocked_handoff",
                                   "timing": "not_needed", "owner_clear": True, "deadline": None}]}
        provider = ScriptedAgent([{"facts": []}, assessment], agent_version=11)
        result, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [], provider, scope)
        self.assertEqual(result.verdict, "insufficient_evidence")

    def test_v12_checks_who_acts_whether_the_promise_is_due_and_whether_it_still_matters(self):
        cases = (
            ("Try resetting it and tell us the result.", "It works now.",
             {"actor": "customer"}, "not_applicable", []),
            ("Send us your account number and we will call you.", "I reached someone using the phone menu.",
             {"conditional": True, "trigger_id": "e2"}, "insufficient_evidence", ["condition_unverified"]),
            ("Send us your account number and we will call you.", "I've sent the account number.",
             {"conditional": True, "trigger_id": "e3"}, "partial", ["pending"]),
            ("I will investigate why the service is down.", "The service is working again.",
             {"relevance_ended_id": "e3"}, "pass", ["no_longer_needed"]),
            ("I will investigate why the service is down.", "Have you checked it yet?",
             {"relevance_ended_id": "e1"}, "partial", ["pending"]),
        )
        for promise, followup, fields, expected, statuses in cases:
            with self.subTest(promise=promise, followup=followup):
                value, scope = ticket(messages=[
                    {"id": "m1", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": "2026-11-13T15:00:00Z", "text": "My service is down."},
                    {"id": "m2", "actor_id": "a", "role": "agent", "author": "Support",
                     "at": "2026-11-13T16:00:00Z", "text": promise},
                    {"id": "m3", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": "2026-11-13T17:00:00Z", "text": followup},
                ]), Scope()
                action = {"reply_id": "e2", "status_ids": ["e2"], "action": promise,
                          "conditional": False, "trigger_id": None, "status": "pending",
                          "timing": "missing_material", "owner_clear": True, "deadline": None,
                          "actor": "support", "relevance_ended_id": None, **fields}
                provider = ScriptedAgent([{"facts": []}, {
                    "applicable": True, "explanation": promise, "actions": [action],
                }], agent_version=12)
                result, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [],
                                        provider, scope)
                self.assertEqual(result.verdict, expected)
                self.assertEqual(provider.trace["checks"]["ownership"]["ownership_validated_statuses"], statuses)
                if expected == "pass":
                    self.assertEqual([ev.ref for ev in result.evidence], ["m2", "m3"])

    def test_v13_separates_a_next_step_a_promise_and_the_issue_outcome(self):
        for reply, later, overrides, confirmation, expected, status in (
            ("Join our secure chat so we can check the charges.", "I'll try later.",
             {"conditional": True}, None, "pass", "clear_step"),
            ("Reset it and let us know the result.", "I will do that.",
             {"actor": "customer"}, None, "not_applicable", None),
            ("We will contact you soon about the charges.", "Someone contacted me. The charges are still there.",
             {"promise": "kept", "promise_evidence_id": "e3", "timing": "missing_material",
              "issue_status": "open", "issue_evidence_id": "e3"}, None, "partial", "fulfilled"),
            ("We will look into the charges.", "Someone contacted me. The charges are still there.",
             {"promise": "unverified", "issue_status": "resolved", "issue_evidence_id": "e3"},
             [], "insufficient_evidence", "unknown"),
            ("We will investigate why these charges are appearing.", "The charges have been reversed.",
             {"promise": "unverified", "issue_status": "resolved", "issue_evidence_id": "e3"},
             [0], "pass", "no_longer_needed"),
            ("Send the bill and then we'll call.", "No call so far.",
             {"promise": "kept", "promise_evidence_id": "e2", "conditional": True, "trigger_id": "e2"},
             None, "insufficient_evidence", "condition_unverified"),
        ):
            with self.subTest(reply=reply, later=later):
                value, scope = ticket(messages=[
                    {"id": "m1", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": "2026-11-13T15:00:00Z", "text": "There are charges I don't recognize."},
                    {"id": "m2", "actor_id": "a", "role": "agent", "author": "Support",
                     "at": "2026-11-13T16:00:00Z", "text": reply},
                    {"id": "m3", "actor_id": "c", "role": "customer", "author": "Customer",
                     "at": "2026-11-13T17:00:00Z", "text": later},
                ]), Scope()
                check = {"reply_id": "e2", "issue": "Unexpected charges", "action": reply, "actor": "support",
                         "next_step": "clear", "timing": "not_needed", "promise": "none",
                         "promise_evidence_id": None, "conditional": False, "trigger_id": None,
                         "deadline": None, "issue_status": "unknown", "issue_evidence_id": None, **overrides}
                replies = [{"facts": []}, {"checks": [check]}]
                if confirmation is not None:
                    replies.append({"confirmed": confirmation})
                provider = ScriptedAgent(replies, agent_version=13)
                result, _ = model_check(criterion("ownership"), value, scoped_messages(value, scope), [],
                                        provider, scope)
                trace = provider.trace["checks"]["ownership"]
                self.assertEqual(result.verdict, expected)
                self.assertEqual(trace["ownership_validated_statuses"], [status] if status else [])
                self.assertEqual(len(provider.requests), 2 + (confirmation is not None))
                if confirmation == [0]:
                    self.assertEqual([ev.ref for ev in result.evidence], ["m2", "m3"])

    def test_quality_checks_unknown_outcome_against_original_sources(self):
        first = {"status":"unknown","evidence_ids":[],"chronology":"A follow-up reply is needed.","explanation":"The outcome is unclear."}
        quote = {"type":"message","ref":"m2","quote":"The refund was issued."}
        for candidate_quote, expected in [(quote,"solution_proposed"), ({**quote,"quote":"This text does not exist."},"unknown")]:
            review = {"status":"solution_proposed","evidence":[candidate_quote],"explanation":"Support proposed a solution."}
            provider = ScriptedAgent([{"facts":[]}, first, review], quality=True)
            sources = [{"type":"message","ref":m.id,"text":m.text,"role":m.role,"at":m.at.isoformat()} for m in ticket().messages]
            result,_ = provider.assess_outcome({"sources":sources})
            self.assertEqual(result.status,expected)
            self.assertEqual(provider.requests[2],{"sources":sources})
            self.assertIsNone(provider.calls[2][1])  # Inherits the provider's thinking setting.

import json
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from qa.api import app
from qa.contracts import Scope
from qa.db import Base, Outbox, Revision, Run, Session, engine
from qa.jobs import evaluate_run
from qa.model_settings import ModelProfile
from qa.service import enqueue, ingest
from test_evaluation import ticket


class ModelSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(engine)

    def setUp(self):
        self.org = "model-profile-" + str(uuid4())
        with_org = patch("qa.api.ORG", self.org)
        with_org.start()
        self.addCleanup(with_org.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_cloud_submission_snapshots_provider_and_worker_uses_server_key(self):
        cloud = {"provider": "openrouter", "model": "example/testing", "output_tokens": 2048}
        payload = {"mode": "cloud", "cloud": cloud, "api_key": "test-credential"}
        saved = self.client.put("/api/settings/model", json=payload)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["configured_cloud_providers"], ["openrouter"])
        settings = self.client.get("/api/settings").json()
        self.assertEqual(settings["model"], cloud["model"])
        self.assertNotIn(payload["api_key"], json.dumps(settings))
        self.assertNotIn(payload["api_key"], saved.text)
        with Session.begin() as session:
            revision = ingest(session, self.org, ticket())
            run_id = enqueue(session, self.org, revision, Scope())
            self.assertEqual(session.get(Run, run_id).config["cloud_provider"], "openrouter")
        payload = {"mode": "local", "local": {
            "model": "local-test", "output_tokens": 1024, "context_tokens": 8192,
            "reasoning_tokens": 768, "agent_mode": True, "agent_quality": True,
        }, "cloud": cloud}
        self.assertEqual(self.client.put("/api/settings/model", json=payload).status_code, 200)
        requests = []

        def respond(url, *, headers, **options):
            payload = options["json"]
            requests.append((url, headers, payload))
            body = {"explanation": "History is insufficient.", "evidence": []}
            schema = json.loads(payload["messages"][0]["content"].split("conforming to this schema: ", 1)[1])
            if "status" in schema["properties"]:
                body["status"] = "unknown"
            else:
                body["verdict"] = "insufficient_evidence"
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "choices": [{"message": {"content": json.dumps(body)}}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 12},
            })

        with patch("qa.cloud.httpx.post", side_effect=respond):
            evaluate_run.run(run_id)
        self.assertTrue(requests)
        for url, headers, body in requests:
            self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(headers["Authorization"], "Bearer test-credential")
            self.assertEqual(body["model"], cloud["model"])
            self.assertEqual(body["max_completion_tokens"], cloud["output_tokens"])
        with Session.begin() as session:
            run = session.get(Run, run_id)
            self.assertEqual(run.status, "complete")
            self.assertNotIn("test-credential", json.dumps(run.config))
            self.assertIsNone(session.get(Outbox, run_id))
            new_id = enqueue(session, self.org, session.get(Revision, run.revision_id), Scope())
            new = session.get(Run, new_id).config
            self.assertNotIn("cloud_provider", new)
            self.assertEqual(new["model"], "local-test")
            self.assertEqual(new["context_tokens"], 8192)
            self.assertEqual(new["reasoning_tokens"], 768)
            self.assertTrue(new["agent_quality"])

    def test_connection_checks_do_not_save_credentials_or_echo_invalid_payloads(self):
        payload = {
            "mode": "cloud", "cloud": {"model": "test-model"},
            "api_key": "temporary-secret",
        }
        with patch("qa.cloud.httpx.post") as call:
            call.return_value = httpx.Response(200, request=httpx.Request("POST", "https://api.openai.com"), json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
            })
            probe = self.client.post("/api/settings/model/test", json=payload)
        self.assertEqual(probe.json(), {"ok": True})
        self.assertEqual(call.call_args.kwargs["json"]["messages"][-1]["content"], "{}")
        self.assertEqual(self.client.get("/api/settings").json()["evaluation"]["configured_cloud_providers"], [])
        response = self.client.put("/api/settings/model", json=[payload])
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(payload["api_key"], response.text)
        invalid = self.client.put("/api/settings/model", json={"mode": "cloud", "cloud": {"model": "test-model"}})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.client.get("/api/settings").json()["evaluation"]["mode"], ModelProfile().mode)

    def test_native_cloud_providers_probe_and_evaluate_with_separate_server_keys(self):
        providers = (
            ("anthropic", "claude-test", "claude-key", "https://api.anthropic.com/v1/messages"),
            ("google", "models/gemini-test", "gemini-key",
             "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"),
        )
        case = ticket()
        with Session.begin() as session:
            revision_id = ingest(session, self.org, case).id
        runs = []
        for provider, model, key, endpoint in providers:
            with self.subTest(provider=provider):
                def respond(url, *, headers, json, **options):
                    self.assertEqual(url, endpoint)
                    if provider == "anthropic":
                        self.assertEqual(headers["x-api-key"], key)
                        self.assertIn("anthropic-version", headers)
                        self.assertEqual(json["model"], model)
                        self.assertEqual(json["max_tokens"], 2048)
                        schema = json["output_config"]["format"]["schema"]
                        self.assertEqual(json["output_config"]["format"]["type"], "json_schema")
                        self.assertIn("properties", schema)
                        self.assertEqual(schema["additionalProperties"], False)
                    else:
                        self.assertEqual(headers["x-goog-api-key"], key)
                        options = json["generationConfig"]
                        self.assertEqual(options["maxOutputTokens"], 2048)
                        self.assertEqual(options["responseMimeType"], "application/json")
                        schema = options["responseJsonSchema"]
                    self.assertNotIn(key, str(json))
                    properties = schema["properties"]
                    if "ok" in properties:
                        text = '{"ok":true}'
                    else:
                        answer = '"status":"unknown"' if "status" in properties else '"verdict":"insufficient_evidence"'
                        text = '{' + answer + ',"evidence":[],"explanation":"History is insufficient."}'
                    if provider == "anthropic":
                        body = {"content": [{"type": "thinking", "thinking": "private analysis"},
                                            {"type": "text", "text": text}],
                                "usage": {"input_tokens": 15, "cache_read_input_tokens": 7, "output_tokens": 12}}
                    else:
                        body = {"candidates": [{"content": {"parts": [{"text": "private analysis", "thought": True},
                                                                      {"text": text}]}}],
                                "usageMetadata": {"promptTokenCount": 22, "cachedContentTokenCount": 7,
                                                  "candidatesTokenCount": 9, "thoughtsTokenCount": 3}}
                    return httpx.Response(200, request=httpx.Request("POST", url), json=body)

                payload = {"mode": "cloud", "cloud": {"provider": provider, "model": model}, "api_key": key}
                with patch("qa.cloud.httpx.post", side_effect=respond) as post:
                    probe = self.client.post("/api/settings/model/test", json=payload)
                    self.assertEqual(probe.json(), {"ok": True})
                    saved = self.client.put("/api/settings/model", json=payload)
                    self.assertEqual(saved.status_code, 200, saved.text)
                    self.assertEqual(saved.json()["cloud"]["model"], model.removeprefix("models/"))
                    self.assertNotIn(key, saved.text)
                    probe = self.client.post("/api/settings/model/test", json={k: v for k, v in payload.items() if k != "api_key"})
                    self.assertEqual(probe.json(), {"ok": True})
                    with Session.begin() as session:
                        run_id = enqueue(session, self.org, session.get(Revision, revision_id), Scope())
                        config = session.get(Run, run_id).config
                        self.assertEqual(config["cloud_provider"], provider)
                        self.assertNotIn(key, json.dumps(config))
                    evaluate_run.run(run_id)
                    self.assertGreater(post.call_count, 2)
                runs.append(run_id)
        self.assertEqual(len(set(runs)), 2)
        for run_id in runs:
            with Session() as session:
                run = session.get(Run, run_id)
                self.assertEqual(run.status, "complete")
                self.assertGreater(run.result["usage"]["prompt_eval_count"], 0)
                self.assertGreater(run.result["usage"]["eval_count"], 0)
                self.assertGreater(run.result["usage"]["prompt_eval_cached_count"], 0)
        configured = self.client.get("/api/settings").json()["evaluation"]["configured_cloud_providers"]
        self.assertEqual(configured, ["anthropic", "google"])

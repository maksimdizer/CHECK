"""Exercise the Zendesk API contract and checkpoint against an isolated PostgreSQL database."""

import json
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from sqlalchemy import delete, func, select

from qa import settings
from qa.db import Base, Conversation, Outbox, Revision, Run, Session, SyncCursor, engine
from qa.zendesk import EXPORT_PATH, ZendeskClient, ZendeskError, ZendeskSettings, sync_page


def comment(number, author, text, *, public=True):
    return {
        "id": number, "author_id": author, "public": public,
        "created_at": f"2025-05-01T12:{number:02d}:00Z", "plain_body": text,
    }


class ZendeskSyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(engine)

    def setUp(self):
        self.org = f"test-zendesk-{uuid4()}"
        self.settings = ZendeskSettings("example", "test-client", "test-secret", 1746000000)
        self.requests = []
        self.reply = "The link was sent."
        self.fail_second_ticket = True
        self.rate_limited = False
        self.first = {"id": 11, "status": "closed", "requester_id": 101, "subject": "Email"}
        self.second = {**self.first, "id": 12, "subject": "Question"}
        self.http = httpx.Client(
            base_url="https://example.zendesk.com", transport=httpx.MockTransport(self.respond),
        )
        self.addCleanup(self.http.close)
        self.addCleanup(self.remove_rows)
        self.client = ZendeskClient(self.settings, self.http)

    def respond(self, request):
        path, params = request.url.path, request.url.params
        self.requests.append((path, dict(params)))
        self.assertEqual(request.url.host, "example.zendesk.com")
        if path == "/oauth/tokens":
            self.assertEqual(request.method, "POST")
            payload = json.loads(request.content)
            self.assertEqual(payload["grant_type"], "client_credentials")
            self.assertEqual(payload["scope"], "tickets:read users:read")
            self.assertEqual(payload["expires_in"], 1800)
            return httpx.Response(200, json={"access_token": "test-bearer", "expires_in": 1800})
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.headers["Authorization"], "Bearer test-bearer")
        if path == EXPORT_PATH:
            self.assertEqual(params["support_type_scope"], "all")
            cursor = params.get("cursor")
            if cursor is None:
                self.assertEqual(params["start_time"], str(self.settings.start_time))
                tickets = [self.first, self.first, {"id": 13, "status": "solved"}, {"id": 14, "status": "open"}]
                next_cursor, done = "first-page", False
            elif cursor == "first-page":
                self.assertNotIn("start_time", params)
                tickets, next_cursor, done = [self.first, self.second], "end", True
            else:
                self.assertEqual(cursor, "end")
                tickets, next_cursor, done = [self.first], "end", True
            return httpx.Response(200, json={"tickets": tickets, "after_cursor": next_cursor, "end_of_stream": done})
        if path == "/api/v2/tickets/12/comments" and self.fail_second_ticket:
            return httpx.Response(503)
        if path in {"/api/v2/tickets/11/comments", "/api/v2/tickets/12/comments"}:
            if "page[after]" not in params:
                if not self.rate_limited:
                    self.rate_limited = True
                    return httpx.Response(429, headers={"Retry-After": "3"})
                return httpx.Response(200, json={
                    "comments": [comment(1, 101, "I did not receive the email.")],
                    "meta": {"has_more": True, "after_cursor": "next-comment"},
                    "links": {"next": "https://elsewhere.invalid/never-follow"},
                })
            self.assertEqual(params["page[after]"], "next-comment")
            return httpx.Response(200, json={
                "comments": [comment(2, 201, "Internal note", public=False), comment(3, 201, self.reply)],
                "meta": {"has_more": False},
            })
        if path == "/api/v2/users/show_many":
            self.assertEqual(params["ids"], "201")
            return httpx.Response(200, json={"users": [{"id": 201, "role": "agent"}]})
        self.fail(f"Unexpected endpoint {path}")

    def remove_rows(self):
        with Session.begin() as session:
            runs = select(Run.id).where(Run.org == self.org)
            conversations = select(Conversation.id).where(Conversation.org == self.org)
            session.execute(delete(Outbox).where(Outbox.run_id.in_(runs)))
            session.execute(delete(Run).where(Run.org == self.org))
            session.execute(delete(Revision).where(Revision.conversation_id.in_(conversations)))
            session.execute(delete(Conversation).where(Conversation.org == self.org))
            session.execute(delete(SyncCursor).where(SyncCursor.org == self.org))

    def snapshot(self):
        with Session() as session:
            state = session.get(SyncCursor, (self.org, self.settings.source))
            runs = list(session.scalars(select(Run).where(Run.org == self.org).order_by(Run.created_at)))
            revisions = [session.get(Revision, run.revision_id).data for run in runs]
            outbox = session.scalar(select(func.count()).select_from(Outbox).where(
                Outbox.run_id.in_(select(Run.id).where(Run.org == self.org)),
            ))
            return state.cursor, revisions, outbox

    def test_stale_response_cannot_rewind_checkpoint(self):
        export = self.client.export

        def race(start_time, cursor):
            page = export(start_time, cursor)
            with patch.object(self.client, "export", side_effect=export):
                self.assertFalse(sync_page(self.client, self.org))
                self.assertTrue(sync_page(self.client, self.org))
            return page

        self.rate_limited = True
        self.fail_second_ticket = False
        with patch.object(self.client, "export", side_effect=race):
            self.assertFalse(sync_page(self.client, self.org))
        cursor, revisions, outbox = self.snapshot()
        self.assertEqual((cursor, outbox), ("end", 2))
        self.assertEqual([ticket["external_id"] for ticket in revisions], ["11", "12"])

    @patch("qa.zendesk.sleep")
    def test_sync_replays_without_duplicates_and_preserves_cursor_on_failure(self, sleep):
        self.assertFalse(sync_page(self.client, self.org))
        sleep.assert_called_once_with(3)
        cursor, revisions, outbox = self.snapshot()
        self.assertEqual((cursor, outbox), ("first-page", 1))
        self.assertEqual([ticket["external_id"] for ticket in revisions], ["11"])
        self.assertEqual([item["role"] for item in revisions[0]["messages"]], ["customer", "agent", "agent"])
        self.assertFalse(revisions[0]["messages"][1]["public"])
        self.assertTrue(revisions[0]["messages_complete"])

        with self.assertRaisesRegex(ZendeskError, "Ticket 12: HTTP 503"):
            sync_page(self.client, self.org)
        self.assertEqual(self.snapshot(), (cursor, revisions, outbox))

        self.fail_second_ticket = False
        self.client = ZendeskClient(self.settings, self.http)
        with patch.object(settings, "MODEL", "changed-model"):
            self.assertTrue(sync_page(self.client, self.org))
            cursor, revisions, outbox = self.snapshot()
            self.assertEqual((cursor, outbox), ("end", 2))
            self.assertEqual([ticket["external_id"] for ticket in revisions], ["11", "12"])
            self.assertTrue(sync_page(self.client, self.org))
            self.assertEqual(self.snapshot(), (cursor, revisions, outbox))

        self.reply = "Sent again."
        with patch("qa.zendesk.enqueue", side_effect=RuntimeError("database write failed")):
            with self.assertRaises(RuntimeError):
                sync_page(self.client, self.org)
        self.assertEqual(self.snapshot(), (cursor, revisions, outbox))
        self.assertTrue(sync_page(self.client, self.org))
        cursor, revisions, outbox = self.snapshot()
        self.assertEqual((cursor, outbox), ("end", 3))
        self.assertEqual(revisions[-1]["messages"][-1]["text"], self.reply)
        self.assertEqual(sum(path == "/oauth/tokens" for path, _ in self.requests), 2)


if __name__ == "__main__":
    unittest.main()

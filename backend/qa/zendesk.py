"""Read-only ticket synchronization from a Zendesk Support account."""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import batched
from time import monotonic, sleep

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from .contracts import Scope, Ticket
from .db import Session, SyncCursor
from .service import enqueue, ingest
from .settings import ORG

logger = logging.getLogger(__name__)
EXPORT_PATH = "/api/v2/incremental/tickets/cursor"
MESSAGE_LIMIT = 300
TEXT_LIMIT = 16000
MAX_PAGES_PER_POLL = 10


class ZendeskError(Exception):
    """A Zendesk failure whose message is safe to put in the service log."""


class TicketUnavailable(ZendeskError):
    pass


@dataclass(frozen=True)
class ZendeskSettings:
    subdomain: str
    client_id: str
    client_secret: str = field(repr=False)
    start_time: int
    interval: int = 120
    language: str = "en"

    @property
    def source(self):
        return f"zendesk:{self.subdomain}"

    @classmethod
    def from_env(cls):
        subdomain = os.getenv("ZENDESK_SUBDOMAIN", "").strip().lower()
        client_id = os.getenv("ZENDESK_CLIENT_ID", "").strip()
        client_secret = os.getenv("ZENDESK_CLIENT_SECRET", "")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", subdomain) or not client_id or not client_secret:
            raise ZendeskError("Set ZENDESK_SUBDOMAIN, ZENDESK_CLIENT_ID and ZENDESK_CLIENT_SECRET.")
        start = os.getenv("ZENDESK_START_AT", "").strip()
        try:
            timestamp = datetime.fromisoformat(start) if start else None
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError
            start_time = int(timestamp.timestamp() if timestamp else datetime.now(timezone.utc).timestamp() - 86400)
            if start_time < 0 or start_time > datetime.now(timezone.utc).timestamp() - 60:
                raise ValueError
        except (ValueError, OverflowError) as exc:
            raise ZendeskError("ZENDESK_START_AT must be a timezone-aware ISO date at least one minute in the past.") from exc
        try:
            interval = int(os.getenv("ZENDESK_POLL_SECONDS", "120"))
            if interval < 60:
                raise ValueError
        except ValueError as exc:
            raise ZendeskError("ZENDESK_POLL_SECONDS must be an integer of at least 60.") from exc
        language = os.getenv("ZENDESK_LANGUAGE", "en").strip()
        if not 1 <= len(language) <= 120:
            raise ZendeskError("ZENDESK_LANGUAGE must contain 1–120 characters.")
        return cls(subdomain, client_id, client_secret, start_time, interval, language)


class ZendeskClient:
    def __init__(self, settings: ZendeskSettings, http: httpx.Client):
        self.settings = settings
        self.http = http
        self.token = ""
        self.token_expiry = 0.0

    def request(self, method, path, **kwargs):
        for attempt in range(4):
            response = self.http.request(method, path, **kwargs)
            if response.status_code != 429 or attempt == 3:
                return response
            try:
                delay = max(1, int(response.headers.get("Retry-After", "60")))
            except ValueError:
                delay = 60
            sleep(delay)

    def authorize(self):
        response = self.request("POST", "/oauth/tokens", json={
            "grant_type": "client_credentials",
            "client_id": self.settings.client_id,
            "client_secret": self.settings.client_secret,
            "scope": "tickets:read users:read",
            "expires_in": 1800,
        })
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        self.token_expiry = monotonic() + max(0, int(data["expires_in"]) - 30)

    def get(self, path, params, *, comments=False):
        if monotonic() >= self.token_expiry:
            self.authorize()
        response = self.request("GET", path, params=params, headers={"Authorization": f"Bearer {self.token}"})
        if response.status_code == 401:
            self.authorize()
            response = self.request("GET", path, params=params, headers={"Authorization": f"Bearer {self.token}"})
        if comments and response.status_code == 404:
            raise TicketUnavailable("Zendesk no longer exposes this ticket's comments.")
        response.raise_for_status()
        return response.json()

    def export(self, start_time, cursor):
        params = {"cursor": cursor} if cursor is not None else {"start_time": start_time}
        data = self.get(EXPORT_PATH, {
            **params, "exclude_deleted": "true", "per_page": 50, "support_type_scope": "all",
        })
        tickets, next_cursor, done = data["tickets"], data["after_cursor"], data["end_of_stream"]
        if (
            not isinstance(tickets, list)
            or not isinstance(done, bool)
            or not isinstance(next_cursor, str)
            or not next_cursor
            or (not done and next_cursor == cursor)
        ):
            raise ZendeskError("Invalid incremental ticket page.")
        return tickets, next_cursor, done

    def comments(self, ticket_id):
        path = f"/api/v2/tickets/{ticket_id}/comments"
        params = {"page[size]": 100, "sort": "created_at"}
        comments = {}
        seen = set()
        while True:
            page = self.get(path, params, comments=True)
            for item in page["comments"]:
                comments[item["id"]] = item
            meta = page["meta"]
            if len(comments) > MESSAGE_LIMIT or (len(comments) == MESSAGE_LIMIT and meta["has_more"]):
                return list(comments.values())[:MESSAGE_LIMIT], False
            if not meta["has_more"]:
                return list(comments.values()), True
            cursor = meta["after_cursor"]
            if not cursor or cursor in seen:
                raise ZendeskError("Invalid comments cursor.")
            seen.add(cursor)
            params["page[after]"] = cursor

    def user_roles(self, ids):
        roles = {}
        for batch in batched(sorted(ids), 100):
            page = self.get("/api/v2/users/show_many", {"ids": ",".join(map(str, batch)), "include_deleted": "true"})
            roles.update({user["id"]: user["role"] for user in page["users"]})
        return roles

    def normalize(self, ticket):
        ticket_id = ticket["id"]
        if type(ticket_id) is not int or ticket_id <= 0:
            raise ZendeskError("Invalid Zendesk ticket ID.")
        try:
            comments, complete = self.comments(ticket_id)
        except TicketUnavailable:
            logger.warning("Zendesk ticket %s skipped: comments returned HTTP 404", ticket_id)
            return None

        requester = ticket.get("requester_id")
        authors = {comment["author_id"] for comment in comments} - {requester, None, -1, 0}
        roles = self.user_roles(authors)
        ai_ticket = ticket.get("support_type") == "ai_agent"
        labels = {"agent": "Agent", "bot": "Bot", "customer": "Customer", "system": "Participant"}
        messages = []
        for comment in comments:
            author = comment["author_id"]
            user_role = roles.get(author)
            if (author is not None and author == requester) or user_role == "end-user":
                role = "customer"
            elif user_role in {"admin", "agent"}:
                role = "bot" if ai_ticket and author == ticket.get("assignee_id") else "agent"
            else:
                role = "system"
                if comment["public"]:
                    complete = False
            body = (comment.get("plain_body") or comment.get("body") or "").strip()
            if not body:
                complete = False
                continue
            if len(body) > TEXT_LIMIT:
                complete = False
            messages.append({
                "id": str(comment["id"]),
                "actor_id": str(author) if author is not None else "unknown",
                "author": f"{labels[role]} {author}" if author is not None else labels[role],
                "role": role,
                "at": comment["created_at"],
                "text": body[:TEXT_LIMIT],
                "public": comment["public"],
            })
        if not messages:
            logger.info("Zendesk ticket %s skipped: no text comments", ticket_id)
            return None
        channel = (ticket.get("via") or {}).get("channel") or "zendesk"
        return Ticket(
            source=self.settings.source,
            external_id=str(ticket_id),
            subject=(ticket.get("subject") or "").strip()[:120] or f"Zendesk #{ticket_id}",
            channel=channel,
            language=self.settings.language,
            status="closed",
            messages_complete=complete,
            messages=messages,
        )


def sync_page(client: ZendeskClient, org: str):
    """Store the checkpoint in the same commit as the page's revisions and outbox."""
    key = {"org": org, "source": client.settings.source}
    with Session.begin() as session:
        session.execute(insert(SyncCursor).values(**key, start_time=client.settings.start_time).on_conflict_do_nothing())
        state = session.get(SyncCursor, (org, key["source"]))
        start_time, cursor = state.start_time, state.cursor

    tickets, next_cursor, done = client.export(start_time, cursor)
    # The stream may repeat a ticket. Only a closed record can be evaluated.
    closed = {item["id"]: item for item in tickets if item["status"] == "closed"}
    revisions = []
    for ticket_id in sorted(closed):
        try:
            ticket = client.normalize(closed[ticket_id])
        except httpx.HTTPStatusError as exc:
            raise ZendeskError(f"Ticket {ticket_id}: HTTP {exc.response.status_code}.") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ZendeskError(f"Ticket {ticket_id}: invalid comment or user data.") from exc
        if ticket is not None:
            revisions.append(ticket)

    with Session.begin() as session:
        state = session.scalar(select(SyncCursor).filter_by(**key).with_for_update())
        if state.cursor != cursor:
            return False
        for ticket in revisions:
            revision = ingest(session, org, ticket)
            enqueue(session, org, revision, Scope(), source_import=True)
        state.cursor = next_cursor
    logger.info("Zendesk page saved: %d closed text ticket(s); caught up=%s", len(revisions), done)
    return done


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = ZendeskSettings.from_env()
    except ZendeskError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None
    with httpx.Client(
        base_url=f"https://{settings.subdomain}.zendesk.com",
        timeout=20, trust_env=False, follow_redirects=False,
    ) as http:
        client = ZendeskClient(settings, http)
        while True:
            delay = settings.interval
            try:
                for _ in range(MAX_PAGES_PER_POLL):
                    if sync_page(client, ORG):
                        break
                    sleep(6)
            except httpx.HTTPStatusError as exc:
                logger.warning("Zendesk sync paused: HTTP %s; checkpoint preserved", exc.response.status_code)
                if exc.response.status_code == 429:
                    try:
                        delay = max(delay, int(exc.response.headers.get("Retry-After", "0")))
                    except ValueError:
                        pass
            except ZendeskError as exc:
                logger.warning("Zendesk sync paused: %s Checkpoint preserved.", exc)
            except Exception as exc:
                logger.warning("Zendesk sync paused (%s); checkpoint preserved", type(exc).__name__)
            sleep(delay)


if __name__ == "__main__":
    main()

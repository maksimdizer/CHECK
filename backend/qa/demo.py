from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text

from .contracts import Scope, Ticket
from .db import Conversation
from .service import enqueue, ingest
from .settings import ORG


def seed(session):
    session.execute(text("SELECT pg_advisory_xact_lock(21030741)"))
    if session.scalar(select(Conversation.id).where(Conversation.org == ORG, Conversation.source == "demo").limit(1)):
        return
    demo = [
        ("1046", "Change a delivery address", "delivery", "Can I change the delivery address?",
         "Yes. Send me the new address and I’ll check whether your order has already shipped.", "Avery Morgan", 4, None),
        ("1047", "Password reset link", "access", "The password reset link won’t open.",
         "I sent you a new link in a separate email. Open it and set a new password.", "Jordan Ellis", 7, None),
        ("1048", "Subscription refund", "refund", "Hi! I canceled my subscription but was charged today. Can I get a refund?",
         "Hi! I checked the subscription charge and issued a refund. The money will return to your original card within 3–5 business days.", "Maria Kim", 22, False),
        ("1049", "Duplicate charge refund", "refund", "I was charged twice for one order.",
         "I refunded the duplicate charge. You can check the refund status under Payments.", "Support bot", 1, True),
    ]
    start = datetime(2026, 11, 12, 15, 16, tzinfo=timezone.utc)
    for number, subject, category, question, response, author, minutes, event in demo:
        reply_at = start + timedelta(minutes=minutes)
        actor_id = author.lower().replace(" ", "-")
        messages = [
            {"id": "m1", "actor_id": "customer", "author": "Customer", "role": "customer", "at": start, "text": question},
            {"id": "m2", "actor_id": actor_id, "author": author, "role": "bot" if event else "agent", "at": reply_at, "text": response},
        ]
        if number == "1048":
            messages.append({"id": "m3", "actor_id": "customer", "author": "Customer", "role": "customer",
                             "at": reply_at + timedelta(minutes=2), "text": "Got it, thanks! I’ll wait for the refund."})
        events = [{"id": "e1", "actor_id": actor_id, "kind": "refund_completed", "at": reply_at, "detail": "The system confirmed the duplicate charge refund."}] if event else []
        ticket = Ticket(
            external_id=number, subject=subject, source="demo", category=category, language="en", messages=messages,
            events=events, messages_complete=True, events_complete=event is not None,
            refund={"decision": "approved", "owner_id": actor_id, "completion_claim_message_id": "m2"} if event is not None else None,
        )
        enqueue(session, ORG, ingest(session, ORG, ticket), Scope(), demo=True)

import csv
import io
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from .contracts import KnowledgeInput, Scope, Ticket
from .db import Conversation, Knowledge, KnowledgeIndex, Outbox, Revision, Run
from .evaluation import retrieve_knowledge, scoped_messages
from .model_settings import active_profile
from .scorecards import active_card
from .settings import AGENT_VERSION, CARD, ENGINE_VERSION, PROMPT_VERSION, SEARCH_VERSION, digest


def import_csv(contents):
    reader = csv.DictReader(io.StringIO(contents, newline=""), strict=True)
    required = {"external_id", "subject", "message_id", "actor_id", "author", "role", "at", "text"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Required CSV columns: " + ", ".join(sorted(required)))
    grouped = {}
    for row in reader:
        if None in row or any(row[field] is None for field in required):
            raise ValueError(f"Invalid number of CSV columns on line {reader.line_num}")

        def flag(key, default):
            value = (row.get(key) or default).strip().lower()
            if value not in {"true", "false"}:
                raise ValueError(f"Column {key} must contain true or false")
            return value == "true"

        data = {key: row[key] for key in ("external_id", "subject")}
        for key in ("source", "channel", "language", "product", "category", "status", "created_at", "closed_at"):
            if row.get(key):
                data[key] = row[key]
        data["messages_complete"] = flag("messages_complete", "false")
        key = (data.get("source", "import").strip(), data["external_id"].strip())
        messages, metadata = grouped.setdefault(key, ([], data))
        if metadata != data:
            raise ValueError(f"Rows for ticket {key[1]} contain inconsistent attributes")
        messages.append({
            "id": row["message_id"],
            **{field: row[field] for field in ("actor_id", "author", "role", "at", "text")},
            "public": flag("public", "true"),
        })
    return [Ticket.model_validate({**data, "messages": messages}) for messages, data in grouped.values()]


def ingest(session, org, ticket):
    key = {"org": org, "source": ticket.source, "external_id": ticket.external_id}
    session.execute(insert(Conversation).values(**key).on_conflict_do_nothing())
    conversation = session.scalar(select(Conversation).filter_by(**key))
    data = ticket.model_dump(mode="json")
    for field in ("created_at", "closed_at"):
        if data[field] is None:
            del data[field]
    key = {"conversation_id": conversation.id, "digest": digest(data)}
    session.execute(insert(Revision).values(**key, data=data).on_conflict_do_nothing())
    return session.scalar(select(Revision).filter_by(**key))


def knowledge_for(session, org, ticket, scope):
    documents = select(func.json_agg(Knowledge.data)).where(Knowledge.org == org).scalar_subquery()
    routes = select(KnowledgeIndex.sections).where(KnowledgeIndex.org == org).scalar_subquery()
    records, sections = session.execute(select(documents, routes)).one()
    return retrieve_knowledge(ticket, scope, [KnowledgeInput.model_validate(doc) for doc in records or []],
                              sections or ())


def enqueue(session, org, revision, scope: Scope, *, force=False, demo=False, source_import=False):
    ticket = Ticket.model_validate(revision.data)
    if ticket.status != "closed":
        raise ValueError("Only closed tickets can be evaluated")
    if scope.kind != "conversation" and not scoped_messages(ticket, scope):
        raise ValueError("The evaluation scope must include a public reply from an agent or bot")
    profile = active_profile(session, org)
    config = {
        "scope": scope.model_dump(mode="json"),
        "card": active_card(session, org),
        "model": "" if demo else profile.model,
        "prompt_version": PROMPT_VERSION,
        "engine_version": ENGINE_VERSION,
        "search_version": SEARCH_VERSION,
        "knowledge": knowledge_for(session, org, ticket, scope),
        "request_outcomes_version": 1,
    }
    if config["model"] and profile.mode == "cloud":
        config["cloud_provider"] = profile.cloud.provider
        config["num_predict"] = profile.cloud.output_tokens
    elif config["model"]:
        local = profile.local
        config["num_predict"] = local.output_tokens
        config.update(local.model_dump(include={"context_tokens", "reasoning_tokens"}, exclude_none=True))
        if local.agent_mode:
            config["agent_mode"] = True
            config["agent_version"] = AGENT_VERSION
            if local.agent_quality:
                config["agent_quality"] = True
        if local.think is not None:
            config["think"] = local.think
    if force:
        config["experiment"] = str(uuid4())
    # An imported revision is scheduled once. Explicit reruns use the current configuration.
    fingerprint = (
        digest([org, revision.id, "source_import"])
        if source_import else digest([org, revision.id, config])
    )
    run_id = session.scalar(insert(Run).values(
        org=org, revision_id=revision.id, fingerprint=fingerprint, config=config,
    ).on_conflict_do_nothing().returning(Run.id))
    if run_id is None:
        return session.scalar(select(Run.id).where(Run.fingerprint == fingerprint))
    session.add(Outbox(run_id=run_id))
    return run_id

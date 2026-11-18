import csv
import json
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert

from .contracts import CardInput, Contract, KnowledgeBundleInput, KnowledgeInput, MarkdownContent, MarkdownKnowledgeInput, ReviewInput, Scope, Ticket
from .cloud import CloudChatProvider
from .db import Base, Conversation, Knowledge, KnowledgeIndex, Review, Revision, Run, Session, engine
from .demo import seed
from .evaluation import OllamaProvider, assessment_metrics, scoped_messages
from .knowledge import markdown_fragments, parse_bundle
from .model_settings import ModelProfileInput, active_profile, public_profile, resolve_cloud_key, save_profile
from .service import enqueue, import_csv, ingest, knowledge_for
from .settings import CARD, DEMO, ORG, TOKEN
from .scorecards import CardConflict, active_card, english_card, save_card

SAMPLE_PREFIXES = {f"{number:02x}" for number in range(0, 256, 10)}
MAX_IMPORT = 5 * 1024 * 1024
ticket_list = TypeAdapter(Annotated[list[Ticket], Field(min_length=1, max_length=100)])
submission_schema = TypeAdapter(Ticket | Annotated[list[Ticket], Field(min_length=1, max_length=100)]).json_schema(
    ref_template="#/paths/~1api~1runs/post/requestBody/content/application~1json/schema/$defs/{model}"
)
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app):
    if not DEMO and not TOKEN:
        raise RuntimeError("Set QA_TOKEN before disabling QA_DEMO")
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(21030740)"))
        Base.metadata.create_all(conn)
    if DEMO:
        with Session.begin() as session:
            seed(session)
    yield


app = FastAPI(title="CHECK", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)


def access(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
    if TOKEN and not (credentials and secrets.compare_digest(credentials.credentials.encode(), TOKEN.encode())):
        raise HTTPException(401, "Enter your access token", headers={"WWW-Authenticate": "Bearer"})
    return "Local reviewer"


Reviewer = Annotated[str, Depends(access)]


@app.get("/api/health")
def health():
    with Session() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/settings")
def settings(reviewer: Reviewer):
    with Session() as session:
        profile = active_profile(session, ORG)
        return {"demo": DEMO, "model": profile.model, "organization": ORG, "card": active_card(session, ORG),
                "evaluation": public_profile(session, ORG, profile)}


async def read_model_profile(request):
    if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json":
        raise HTTPException(415, "Send the model profile as application/json.")
    try:
        body = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, "The model profile must be valid JSON.") from exc
    try:
        return ModelProfileInput.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        # Field names in an unexpected JSON object are user input too.
        detail = [{"loc": error["loc"], "msg": error["msg"]} for error in errors if error["type"] != "extra_forbidden"]
        raise HTTPException(422, detail or "The model profile contains unexpected fields.") from exc


@app.put("/api/settings/model")
async def update_model_settings(request: Request, reviewer: Reviewer):
    payload = await read_model_profile(request)

    def update():
        try:
            with Session.begin() as session:
                profile = save_profile(session, ORG, payload)
                session.flush()
                return public_profile(session, ORG, profile)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return await run_in_threadpool(update)


class ModelProbe(Contract):
    ok: Literal[True]


@app.post("/api/settings/model/test")
async def test_model_settings(request: Request, reviewer: Reviewer):
    payload = await read_model_profile(request)

    def probe():
        if not payload.model:
            raise HTTPException(422, "Enter a model ID to test.")
        try:
            if payload.mode == "cloud":
                with Session() as session:
                    api_key = resolve_cloud_key(session, ORG, payload)
                if not api_key:
                    raise HTTPException(422, "Enter an API key to test the connection.")
                provider = CloudChatProvider(payload.model, provider=payload.cloud.provider, api_key=api_key,
                                             num_predict=payload.cloud.output_tokens)
            else:
                local = payload.local
                provider = OllamaProvider(payload.model, num_predict=local.output_tokens, context_tokens=local.context_tokens,
                                          reasoning_tokens=local.reasoning_tokens)
                provider.cache_id = None
            provider._generate({}, ModelProbe, 'This is a connection test. Return only JSON {"ok": true}.', think=False)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                message = "The model server rejected the credentials or permissions."
            elif status == 404:
                message = "The model server could not find the selected model."
            elif status == 429:
                message = "The model server reached a rate or quota limit."
            else:
                message = f"The model server rejected the request (HTTP {status}). Check the model and token limits."
            raise HTTPException(502, message) from exc
        except httpx.RequestError as exc:
            raise HTTPException(502, "Could not reach the model server.") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, "The key or model response is invalid. Check JSON output support.") from exc
        return {"ok": True}

    return await run_in_threadpool(probe)


@app.put("/api/card")
def update_card(payload: CardInput, reviewer: Reviewer):
    try:
        with Session.begin() as session:
            return save_card(session, ORG, payload)
    except CardConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def run_query():
    return (
        select(Run, Revision, Conversation)
        .join(Revision, Revision.id == Run.revision_id)
        .join(Conversation, Conversation.id == Revision.conversation_id)
        .where(Run.org == ORG, Conversation.org == ORG)
    )


def run_summary(run, revision, conversation, reviewed):
    metrics = run.result or {}
    if run.result and "weighted_coverage" not in metrics:
        metrics = assessment_metrics(run.result["criteria"], run.config.get("card", {}).get("groups"))
    return {
        "id": run.id, "revision_id": revision.id, "source": conversation.source,
        "external_id": conversation.external_id, "subject": revision.data["subject"],
        "category": revision.data["category"], "channel": revision.data["channel"],
        "created_at": run.created_at, "status": run.status, "error": run.error,
        "scope": run.config["scope"], "reviewed": reviewed,
        "review_required": bool(run.result and (
            run.result["review_required"] or run.fingerprint[:2] in SAMPLE_PREFIXES
        )),
        "score": run.result["score"] if run.result else None,
        "coverage": run.result["coverage"] if run.result else None,
        "weighted_coverage": metrics.get("weighted_coverage"),
        "checked_weight": metrics.get("checked_weight"),
        "applicable_weight": metrics.get("applicable_weight"),
        "outcome_status": run.result.get("outcome", {}).get("status") if run.result else None,
        "critical": run.result["critical"] if run.result else None,
    }


@app.get("/api/runs")
def list_runs(
    reviewer: Reviewer,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: str = Query("", max_length=120),
    status: Literal["all", "review", "reviewed", "failed"] = "all",
):
    reviewed = select(Review.id).where(Review.run_id == Run.id).exists()
    query = run_query()
    if search.strip():
        query = query.where(or_(
            Conversation.external_id.icontains(search.strip(), autoescape=True),
            Revision.data["subject"].as_string().icontains(search.strip(), autoescape=True),
        ))
    if status == "reviewed":
        query = query.where(reviewed)
    elif status == "failed":
        query = query.where(Run.status == "failed")
    elif status == "review":
        # Sample about 10% of otherwise unflagged runs, using the idempotency hash.
        query = query.where(and_(
            Run.status == "complete",
            ~reviewed,
            or_(Run.result["review_required"].as_boolean(), func.substr(Run.fingerprint, 1, 2).in_(SAMPLE_PREFIXES)),
        ))
    with Session() as session:
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        rows = session.execute(query.add_columns(reviewed).order_by(Run.created_at.desc(), Run.id).offset(offset).limit(limit))
        return {"items": [run_summary(*row) for row in rows], "total": total}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, reviewer: Reviewer):
    with Session() as session:
        row = session.execute(run_query().where(Run.id == run_id)).one_or_none()
        if row is None:
            raise HTTPException(404, "Evaluation not found")
        run, revision, conversation = row
        reviews = list(session.scalars(select(Review).where(Review.run_id == run.id).order_by(Review.created_at, Review.id)))
        card = english_card(run.config["card"])
        return {
            **run_summary(run, revision, conversation, bool(reviews)),
            "ticket": revision.data,
            "result": run.result,
            "reviews": [{"id": item.id, "reviewer": item.reviewer, "created_at": item.created_at, **item.decision} for item in reviews],
            "config": run.config,
            "display_names": {
                "criteria": {item["id"]: item["name"] for item in card["criteria"]},
                "groups": {item["id"]: item["name"] for item in card.get("groups") or []},
            },
            "attempts": run.attempts,
            "finished_at": run.finished_at,
        }


async def read_tickets(request: Request, format: str = "json"):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_IMPORT:
            raise HTTPException(413, "Maximum file size is 5 MB")
        body.extend(chunk)
    try:
        contents = body.decode("utf-8-sig")
        if format == "json":
            data = json.loads(contents)
            tickets = ticket_list.validate_python(data if isinstance(data, list) else [data])
        else:
            tickets = ticket_list.validate_python(import_csv(contents))
    except ValidationError as exc:
        raise HTTPException(422, [
            {"location": list(item["loc"]), "message": item["msg"]}
            for item in exc.errors(include_input=False)
        ]) from exc
    except (ValueError, csv.Error) as exc:
        raise HTTPException(422, f"Could not read the file. {exc}") from exc
    return tickets


class SubmittedRun(Contract):
    external_id: str
    revision_id: str
    run_id: str
    status: Literal["queued", "running", "complete", "failed"]
    status_url: str


class SubmissionResult(Contract):
    items: list[SubmittedRun]


@app.post(
    "/api/runs",
    status_code=202,
    response_model=SubmissionResult,
    responses={413: {"description": "Request exceeds 5 MB"}},
    openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": submission_schema}}}},
)
async def submit_runs(request: Request, response: Response, reviewer: Reviewer):
    """Queue one closed ticket or a batch of up to 100 closed tickets in the import JSON format."""
    tickets = await read_tickets(request)
    opened = [ticket.external_id for ticket in tickets if ticket.status != "closed"]
    if opened:
        raise HTTPException(422, {"message": "Only closed tickets can be evaluated.", "external_ids": opened})
    result = await run_in_threadpool(save_import, tickets, with_status=True)
    if len(result["items"]) == 1:
        response.headers["Location"] = result["items"][0]["status_url"]
    return result


@app.post("/api/import", status_code=202)
async def import_file(request: Request, reviewer: Reviewer, format: str = Query("json", pattern="^(json|csv)$")):
    return await run_in_threadpool(save_import, await read_tickets(request, format))


def save_import(tickets, *, with_status=False):
    # Stable lock ordering prevents deadlocks between overlapping batches.
    ordered = sorted(enumerate(tickets), key=lambda pair: (pair[1].source, pair[1].external_id))
    with Session.begin() as session:
        items = {}
        for index, ticket in ordered:
            revision = ingest(session, ORG, ticket)
            run_id = enqueue(session, ORG, revision, Scope()) if ticket.status == "closed" else None
            item = {"revision_id": revision.id, "run_id": run_id, "external_id": ticket.external_id}
            if with_status:
                item.update(status=session.get(Run, run_id).status, status_url=f"/api/runs/{run_id}")
            items[index] = item
        return {"items": list(items.values()) if not with_status else [items[i] for i in range(len(tickets))]}


class Rerun(Contract):
    scope: Scope = Field(default_factory=Scope)
    force: bool = True


@app.post("/api/runs/{run_id}/rerun", status_code=202)
def rerun(run_id: str, payload: Rerun, reviewer: Reviewer):
    with Session.begin() as session:
        row = session.execute(run_query().where(Run.id == run_id)).one_or_none()
        if row is None:
            raise HTTPException(404, "Evaluation not found")
        try:
            return {"id": enqueue(session, ORG, row[1], payload.scope, force=payload.force)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post("/api/runs/{run_id}/reviews", status_code=201)
def review(run_id: str, payload: ReviewInput, reviewer: Reviewer):
    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.org == ORG, Run.id == run_id).with_for_update())
        if run is None:
            raise HTTPException(404, "Evaluation not found")
        if run.status != "complete":
            raise HTTPException(409, "Wait for the evaluation to finish")
        if payload.action == "correct_outcome" and not run.result.get("outcome"):
            raise HTTPException(422, "This evaluation has no ticket outcome. Run a new evaluation.")
        if payload.criterion_id and payload.criterion_id not in {item["criterion_id"] for item in run.result["criteria"]}:
            raise HTTPException(422, "Criterion not found in this evaluation")
        record = Review(run_id=run.id, reviewer=reviewer, decision=payload.model_dump(mode="json"))
        session.add(record)
        session.flush()
        return {"id": record.id}


@app.post("/api/knowledge", status_code=201)
def add_knowledge(payload: KnowledgeInput, reviewer: Reviewer):
    return save_knowledge(payload)


def save_knowledge(payload):
    with Session.begin() as session:
        revision_id = session.scalar(insert(Knowledge).values(
            org=ORG, document=payload.document, version=payload.version, data=payload.model_dump(mode="json"),
        ).on_conflict_do_nothing().returning(Knowledge.id))
        if revision_id is None:
            raise HTTPException(409, "This document version already exists")
        return {"id": revision_id}


def parse_markdown(markdown):
    try:
        return markdown_fragments(markdown)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/knowledge/markdown/preview")
def preview_markdown(payload: MarkdownContent, reviewer: Reviewer):
    return {"fragments": parse_markdown(payload.markdown)}


@app.post("/api/knowledge/markdown", status_code=201)
def add_markdown_knowledge(payload: MarkdownKnowledgeInput, reviewer: Reviewer):
    fragments = parse_markdown(payload.markdown)
    return save_knowledge(KnowledgeInput(**payload.model_dump(exclude={"markdown"}), fragments=fragments))


async def read_knowledge_bundle(request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_IMPORT:
            raise HTTPException(413, "Knowledge bundles can be up to 5 MB.")
        body.extend(chunk)
    try:
        payload = KnowledgeBundleInput.model_validate_json(body)
        return parse_bundle(payload)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False, include_input=False)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/knowledge/bundle/preview")
async def preview_knowledge_bundle(request: Request, reviewer: Reviewer):
    documents, sections = await read_knowledge_bundle(request)
    return {"sections": sections, "documents": [
        {"path": path, "document": doc.document, "version": doc.version, "fragment_count": len(doc.fragments)}
        for path, doc in documents.items()
    ]}


def publish_knowledge_bundle(documents, sections):
    with Session.begin() as session:
        session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": "knowledge-index:" + ORG})
        for doc in sorted(documents.values(), key=lambda item: (item.document, item.version)):
            contents = doc.model_dump(mode="json")
            inserted = session.scalar(insert(Knowledge).values(
                org=ORG, document=doc.document, version=doc.version, data=contents,
            ).on_conflict_do_nothing().returning(Knowledge.id))
            if not inserted:
                existing = session.scalar(select(Knowledge).where(
                    Knowledge.org == ORG, Knowledge.document == doc.document, Knowledge.version == doc.version,
                ))
                if existing is None or KnowledgeInput.model_validate(existing.data).model_dump(mode="json") != contents:
                    raise HTTPException(409, f"Instruction {doc.document} v{doc.version} already exists with different content.")
        session.execute(insert(KnowledgeIndex).values(org=ORG, sections=sections).on_conflict_do_update(
            index_elements=[KnowledgeIndex.org], set_={"sections": sections},
        ))
    return {"documents": len(documents), "sections": len(sections)}


@app.post("/api/knowledge/bundle", status_code=201)
async def add_knowledge_bundle(request: Request, reviewer: Reviewer):
    documents, sections = await read_knowledge_bundle(request)
    return await run_in_threadpool(publish_knowledge_bundle, documents, sections)


@app.get("/api/knowledge")
def list_knowledge(reviewer: Reviewer, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    with Session() as session:
        query = select(Knowledge).where(Knowledge.org == ORG)
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        documents = session.scalars(query.order_by(Knowledge.document, Knowledge.version.desc())
                                    .offset(offset).limit(limit))
        items = []
        for row in documents:
            document = KnowledgeInput.model_validate(row.data)
            items.append({"id": row.id, **document.model_dump(exclude={"fragments"}),
                          "fragment_count": len(document.fragments)})
        return {"items": items, "total": total}


@app.get("/api/knowledge/preview")
def preview_knowledge(reviewer: Reviewer, run_id: str = Query(min_length=1, max_length=80)):
    with Session() as session:
        row = session.execute(run_query().where(Run.id == run_id)).one_or_none()
        if row is None:
            raise HTTPException(404, "Evaluation not found")
        run, revision, _ = row
        ticket = Ticket.model_validate(revision.data)
        scope = Scope.model_validate(run.config["scope"])
        sources = knowledge_for(session, ORG, ticket, scope)
        return {"run_id": run.id, "subject": ticket.subject, "external_id": ticket.external_id,
                "scope": scope, "items": sources,
                "replies": [{"id": item.id, "author": item.author, "at": item.at, "text": item.text}
                            for item in scoped_messages(ticket, scope)]}

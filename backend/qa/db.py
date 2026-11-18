import os
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import BigInteger, JSON, Text, DateTime, ForeignKey, Index, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("org", "source", "external_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    org: Mapped[str] = mapped_column(String(120), index=True)
    source: Mapped[str] = mapped_column(String(120))
    external_id: Mapped[str] = mapped_column(String(120))


class Revision(Base):
    __tablename__ = "revisions"
    __table_args__ = (UniqueConstraint("conversation_id", "digest"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    digest: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    data: Mapped[dict] = mapped_column(JSON)


class Knowledge(Base):
    __tablename__ = "knowledge"
    __table_args__ = (UniqueConstraint("org", "document", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    org: Mapped[str] = mapped_column(String(120), index=True)
    document: Mapped[str] = mapped_column(String(120))
    version: Mapped[int]
    data: Mapped[dict] = mapped_column(JSON)


class KnowledgeIndex(Base):
    __tablename__ = "knowledge_indexes"
    org: Mapped[str] = mapped_column(String(120), primary_key=True)
    sections: Mapped[list] = mapped_column(JSON)


class Scorecard(Base):
    __tablename__ = "scorecards"
    org: Mapped[str] = mapped_column(String(120), primary_key=True)
    version: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class EvaluationProfile(Base):
    __tablename__ = "evaluation_profiles"
    org: Mapped[str] = mapped_column(String(120), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)


class CloudCredential(Base):
    __tablename__ = "cloud_credentials"
    org: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider: Mapped[str] = mapped_column(String(30), primary_key=True)
    api_key: Mapped[str] = mapped_column(Text)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("run_org_created", "org", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    org: Mapped[str] = mapped_column(String(120))
    revision_id: Mapped[str] = mapped_column(ForeignKey("revisions.id"))
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    lease: Mapped[str | None] = mapped_column(String(36))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Outbox(Base):
    __tablename__ = "outbox"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class SyncCursor(Base):
    __tablename__ = "sync_cursors"
    org: Mapped[str] = mapped_column(String(120), primary_key=True)
    source: Mapped[str] = mapped_column(String(120), primary_key=True)
    start_time: Mapped[int] = mapped_column(BigInteger)
    cursor: Mapped[str | None] = mapped_column(Text)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(120))
    decision: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


engine = create_engine(
    os.getenv("DATABASE_URL", "postgresql+psycopg://qa:qa-local@localhost/qa"),
    pool_pre_ping=True,
    hide_parameters=True,
)
Session = sessionmaker(engine, expire_on_commit=False)

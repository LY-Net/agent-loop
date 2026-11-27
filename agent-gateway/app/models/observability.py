
import uuid

from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    BigInteger,
    DateTime,
    ForeignKey,
    Index
)

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utc_now():
    return datetime.now(timezone.utc)


def generate_id():
    return str(uuid.uuid4())


class AgentTrace(Base):

    __tablename__ = "agent_trace"

    trace_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id
    )

    request_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True
    )

    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True
    )

    session_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="running",
        nullable=False
    )

    input_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False
    )

    output_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    duration_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )


class AgentRun(Base):

    __tablename__ = "agent_run"

    run_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id
    )

    trace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_trace.trace_id"),
        nullable=False,
        index=True
    )

    agent_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="running",
        nullable=False
    )

    input_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False
    )

    output_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    duration_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True
    )


class AgentSpan(Base):

    __tablename__ = "agent_span"

    span_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id
    )

    trace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_trace.trace_id"),
        nullable=False,
        index=True
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_run.run_id"),
        nullable=False,
        index=True
    )

    parent_span_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("agent_span.span_id"),
        nullable=True,
        index=True
    )

    span_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False
    )

    span_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="running",
        nullable=False
    )

    input_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False
    )

    output_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    duration_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    __table_args__ = (
        Index(
            "idx_span_trace_start",
            "trace_id",
            "started_at"
        ),
    )


class AgentEvent(Base):

    __tablename__ = "agent_event"

    event_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id
    )

    trace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_trace.trace_id"),
        nullable=False,
        index=True
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_run.run_id"),
        nullable=False,
        index=True
    )

    span_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("agent_span.span_id"),
        nullable=True,
        index=True
    )

    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False
    )

    event_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

# agent_session table
class AgentSession(Base):

    __tablename__ = "agent_session"

    session_record_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        nullable=False,
        index=True,
    )

    context_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    pending_clarification: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    clarification_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

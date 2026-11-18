from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Verdict = Literal["pass", "partial", "fail", "not_applicable", "insufficient_evidence"]
OutcomeStatus = Literal["resolved", "solution_proposed", "escalated", "unresolved", "unknown"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Message(Contract):
    id: Identifier
    actor_id: Identifier
    author: Identifier
    role: Literal["customer", "agent", "bot", "system"]
    at: AwareDatetime
    text: str = Field(min_length=1, max_length=16000)
    public: bool = True


class Event(Contract):
    id: Identifier
    actor_id: Identifier
    kind: Identifier
    at: AwareDatetime
    detail: str = Field(min_length=1, max_length=4000)
    operation_id: Identifier | None = None


class RefundContext(Contract):
    decision: Literal["approved", "declined", "pending", "not_required", "unknown"] = "unknown"
    due_at: AwareDatetime | None = None
    operation_id: Identifier | None = None
    owner_id: Identifier | None = None
    completion_claim_message_id: Identifier | None = None


class Ticket(Contract):
    external_id: Identifier
    source: Identifier = "import"
    subject: Identifier
    channel: Identifier = "chat"
    language: Identifier = "en"
    product: Identifier = "general"
    category: Identifier = "general"
    status: Literal["closed", "open"] = "closed"
    messages_complete: bool = False
    events_complete: bool = False
    observed_at: AwareDatetime | None = None
    created_at: AwareDatetime | None = None
    closed_at: AwareDatetime | None = None
    refund: RefundContext | None = None
    messages: list[Message] = Field(min_length=1, max_length=300)
    events: list[Event] = Field(default_factory=list, max_length=300)

    @model_validator(mode="after")
    def normalize(self):
        for items in (self.messages, self.events):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Message IDs and event IDs must be unique within their respective groups")
            items.sort(key=lambda item: (item.at, item.id))
        if self.observed_at and any(item.at > self.observed_at for item in [*self.messages, *self.events]):
            raise ValueError("Snapshot time cannot precede any message or event")
        if self.closed_at and (self.status != "closed" or
                               (self.created_at and self.closed_at < self.created_at) or
                               (self.observed_at and self.closed_at > self.observed_at)):
            raise ValueError("Closure time must be within the lifetime of a closed ticket")
        if self.refund and self.refund.completion_claim_message_id:
            claim = next((m for m in self.messages if m.id == self.refund.completion_claim_message_id), None)
            if not claim or not claim.public or claim.role not in {"agent", "bot"}:
                raise ValueError("A refund completion claim must reference a public support reply")
        return self


class Scope(Contract):
    kind: Literal["conversation", "actor", "message"] = "conversation"
    ref: Identifier | None = None

    @model_validator(mode="after")
    def validate_ref(self):
        if (self.kind == "conversation") == (self.ref is not None):
            raise ValueError("An agent or message scope requires ref")
        return self


class KnowledgeMetadata(Contract):
    document: Identifier
    version: int = Field(ge=1)
    title: Identifier
    product: Identifier = "general"
    category: Identifier = "general"
    language: Identifier = "en"
    effective_from: AwareDatetime
    effective_to: AwareDatetime | None = None
    available_from: AwareDatetime
    available_to: AwareDatetime | None = None
    audience: list[Identifier] = Field(default_factory=list, max_length=100)
    @model_validator(mode="after")
    def validate_dates(self):
        if (self.effective_to and self.effective_to <= self.effective_from) or (
            self.available_to and self.available_to <= self.available_from
        ):
            raise ValueError("The end of the period must follow the start")
        return self


class KnowledgeInput(KnowledgeMetadata):
    fragments: dict[Identifier, Annotated[str, StringConstraints(min_length=1, max_length=5000)]] = Field(min_length=1, max_length=30)


class MarkdownContent(Contract):
    markdown: str = Field(min_length=1, max_length=160000)


class MarkdownKnowledgeInput(KnowledgeMetadata, MarkdownContent):
    pass


class BundleKnowledgeInput(MarkdownKnowledgeInput):
    path: str = Field(min_length=1, max_length=400)


class KnowledgeBundleInput(Contract):
    index: str = Field(min_length=1, max_length=160000)
    documents: list[BundleKnowledgeInput] = Field(min_length=1, max_length=100)


class Evidence(Contract):
    type: Literal["message", "event", "knowledge"]
    ref: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=500)


class ModelAnswer(Contract):
    verdict: Verdict
    explanation: str = Field(min_length=1, max_length=1500)
    evidence: list[Evidence] = Field(max_length=12)


class OutcomeAnswer(Contract):
    status: OutcomeStatus
    explanation: str = Field(min_length=1, max_length=1500)
    evidence: list[Evidence] = Field(max_length=12)


class RequestOutcomeEvidence(Contract):
    type: Literal["message", "event"]
    ref: Identifier
    quote: str = Field(min_length=1, max_length=280)


class OutcomeRequest(Contract):
    request: str = Field(min_length=1, max_length=240)
    request_ref: Identifier


class OutcomeRequests(Contract):
    requests: list[OutcomeRequest] = Field(max_length=20)


class RequestOutcome(Contract):
    request_id: int = Field(ge=1, le=20)
    evidence: list[RequestOutcomeEvidence] = Field(max_length=3)
    state: Literal["resolved", "mostly_resolved", "partly_resolved", "remedy_proposed", "unresolved", "unknown"]


class RequestOutcomeAnswer(Contract):
    requests: list[RequestOutcome] = Field(max_length=20)


class ReviewInput(Contract):
    action: Literal["confirm", "correct", "correct_outcome", "rubric_issue", "missing_data"]
    criterion_id: Identifier | None = None
    verdict: Verdict | None = None
    outcome_status: OutcomeStatus | None = None
    comment: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "correct" and not (self.criterion_id and self.verdict and self.comment):
            raise ValueError("A criterion correction requires a criterion, verdict, and comment")
        if self.action != "correct" and self.verdict:
            raise ValueError("Specify a verdict only when correcting a criterion")
        if self.action == "correct_outcome" and (not self.outcome_status or not self.comment or self.criterion_id):
            raise ValueError("An outcome correction requires an outcome status and comment. Leave the criterion empty")
        if self.action != "correct_outcome" and self.outcome_status:
            raise ValueError("Specify an outcome status only when correcting the outcome")
        if self.action in {"rubric_issue", "missing_data"} and not self.comment:
            raise ValueError("Add a comment")
        return self


class CriterionDefinition(Contract):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,79}$")]
    version: int = Field(default=1, ge=1)
    group_id: Identifier
    name: Identifier
    description: str = Field(min_length=1, max_length=3000)
    applicability: str = Field(min_length=1, max_length=1500)
    evaluator_type: Literal["code", "llm", "hybrid"] = "llm"
    required_evidence: list[Literal["message", "event", "knowledge"]] = Field(default_factory=lambda: ["message"], max_length=3)
    rubric: dict[Verdict, str] = Field(min_length=2, max_length=5)
    weight: float = Field(ge=0, le=1000, allow_inf_nan=False)
    severity: Literal["standard", "critical"] = "standard"
    review_policy: Literal["on_fail", "always"] = "on_fail"
    category: Identifier | None = None
    channel_threshold_seconds: dict[Identifier, int] | None = None

    @model_validator(mode="after")
    def validate_definition(self):
        if not {"pass", "fail"}.issubset(self.rubric) or any(not text.strip() or len(text) > 2000 for text in self.rubric.values()):
            raise ValueError("Pass and Fail require descriptions. Each rubric entry is limited to 2,000 characters")
        if self.channel_threshold_seconds and any(not 0 < seconds <= 604800 for seconds in self.channel_threshold_seconds.values()):
            raise ValueError("The channel time target must be between 1 and 604,800 seconds")
        return self


class CriterionGroup(Contract):
    id: Identifier
    name: Identifier
    weight: float | None = Field(default=None, ge=0, le=40000, allow_inf_nan=False)


TimingMetric = Literal["first_response", "time_to_close"]
TimingSeconds = Annotated[int, Field(gt=0, le=31_536_000, strict=True)]


class TimingTarget(Contract):
    default_seconds: TimingSeconds | None = None
    channel_seconds: dict[Identifier, TimingSeconds] = Field(default_factory=dict, max_length=30)


class CardInput(Contract):
    base_version: int = Field(ge=1)
    name: Identifier
    groups: list[CriterionGroup] = Field(min_length=1, max_length=20)
    criteria: list[CriterionDefinition] = Field(min_length=1, max_length=40)
    timing_targets: dict[TimingMetric, TimingTarget] | None = None

    @model_validator(mode="after")
    def validate_card(self):
        groups = {group.id for group in self.groups}
        if len(groups) != len(self.groups) or len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Group IDs and criterion IDs must be unique")
        if any(item.group_id not in groups for item in self.criteria):
            raise ValueError("Each criterion must belong to an existing group")
        if not any(item.weight > 0 and item.id != "first_response" for item in self.criteria):
            raise ValueError("At least one criterion must have a positive weight")
        return self

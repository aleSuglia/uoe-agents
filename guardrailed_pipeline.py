"""Guardrail + Pipeline Design Pattern Example.

Pipeline implemented here (customer support triage):

  Raw ticket
    |
    v
  [Stage 1] Intake validation guardrails
    |
    v
  [Stage 2] Classification + routing guardrails
    |
    v
  [Stage 3] Draft response (or escalate)
    |
    v
  [Stage 4] Final compliance guardrails
    |
    v
  Final decision + audit trail
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Shared model configuration. Swap model_name/provider to use another backend.
ollama_model = OpenAIChatModel(
    model_name="qwen3.5:2b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

ALLOWED_URGENCY = {"low", "medium", "high", "critical"}
ALLOWED_CATEGORIES = {"billing", "technical", "account_access", "refund", "other"}
CONFIDENCE_THRESHOLD = 0.65
BANNED_RESPONSE_PHRASES = {
    "we guarantee",
    "guaranteed",
    "automatic refund",
    "full refund confirmed",
}


class SupportTicket(BaseModel):
    """Input ticket received by the pipeline."""

    ticket_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    urgency: str
    message: str


class Classification(BaseModel):
    """Output of classification stage."""

    category: Literal["billing", "technical", "account_access", "refund", "other"]
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class DraftResponse(BaseModel):
    """Output of response drafting stage."""

    subject: str
    body: str


class GuardrailEvent(BaseModel):
    """Single guardrail check result for auditing/debugging."""

    stage: str
    name: str
    passed: bool
    reason: str


class PipelineResult(BaseModel):
    """Final output of the full pipeline."""

    status: Literal["approved", "escalated", "rejected"]
    ticket_id: str
    escalation_required: bool
    classification: Classification | None = None
    draft_response: DraftResponse | None = None
    audit_trail: list[GuardrailEvent]


classification_agent = Agent(
    ollama_model,
    output_type=Classification,
    system_prompt=(
        "You are a support ticket triage specialist. "
        "Classify the incoming ticket into one category from: "
        "billing, technical, account_access, refund, other. "
        "Also set severity from: low, medium, high, critical. "
        "Return confidence between 0.0 and 1.0 where low certainty means ambiguous input. "
        "Keep rationale short and concrete."
    ),
)


draft_agent = Agent(
    ollama_model,
    output_type=DraftResponse,
    system_prompt=(
        "You are a customer support responder. "
        "Write a concise and polite response with a clear subject line. "
        "Do not promise refunds or guarantees. "
        "Body must include: acknowledgement, one action, and a next step."
    ),
)


def _event(stage: str, name: str, passed: bool, reason: str) -> GuardrailEvent:
    return GuardrailEvent(stage=stage, name=name, passed=passed, reason=reason)


def _validate_intake(ticket: SupportTicket) -> list[GuardrailEvent]:
    events: list[GuardrailEvent] = []

    has_ticket_id = bool(ticket.ticket_id.strip())
    events.append(
        _event(
            stage="intake",
            name="ticket_id_present",
            passed=has_ticket_id,
            reason="ok" if has_ticket_id else "ticket_id is empty",
        )
    )

    valid_urgency = ticket.urgency in ALLOWED_URGENCY
    events.append(
        _event(
            stage="intake",
            name="urgency_allowed",
            passed=valid_urgency,
            reason=(
                "ok"
                if valid_urgency
                else f"urgency must be one of {sorted(ALLOWED_URGENCY)}"
            ),
        )
    )

    msg_len = len(ticket.message.strip())
    long_enough = msg_len >= 20
    events.append(
        _event(
            stage="intake",
            name="message_min_length",
            passed=long_enough,
            reason="ok" if long_enough else "message must be at least 20 characters",
        )
    )

    return events


def _apply_local_confidence_adjustments(
    ticket: SupportTicket, classification: Classification
) -> Classification:
    """Apply a small deterministic adjustment to make demo behavior stable.

    Very short or vague tickets should trend toward lower confidence, even when
    the model returns an optimistic score.
    """
    word_count = len(ticket.message.split())
    adjusted_confidence = classification.confidence

    if word_count < 8:
        adjusted_confidence = min(adjusted_confidence, 0.45)

    if (
        "not sure" in ticket.message.lower()
        or "something is wrong" in ticket.message.lower()
    ):
        adjusted_confidence = min(adjusted_confidence, 0.55)

    if adjusted_confidence == classification.confidence:
        return classification

    return Classification(
        category=classification.category,
        severity=classification.severity,
        confidence=adjusted_confidence,
        rationale=(
            classification.rationale
            + " | Confidence reduced by deterministic guardrail due to ambiguous input."
        ),
    )


def _validate_classification(classification: Classification) -> list[GuardrailEvent]:
    events: list[GuardrailEvent] = []

    allowed_category = classification.category in ALLOWED_CATEGORIES
    events.append(
        _event(
            stage="classification",
            name="category_allowed",
            passed=allowed_category,
            reason="ok" if allowed_category else "category not in allowed taxonomy",
        )
    )

    confidence_ok = classification.confidence >= CONFIDENCE_THRESHOLD
    events.append(
        _event(
            stage="classification",
            name="confidence_threshold",
            passed=confidence_ok,
            reason=(
                "ok"
                if confidence_ok
                else (
                    f"confidence {classification.confidence:.2f} below threshold "
                    f"{CONFIDENCE_THRESHOLD:.2f}; escalate"
                )
            ),
        )
    )

    return events


def _validate_draft(draft: DraftResponse) -> list[GuardrailEvent]:
    events: list[GuardrailEvent] = []

    max_len_ok = len(draft.body) <= 600
    events.append(
        _event(
            stage="draft",
            name="max_length",
            passed=max_len_ok,
            reason="ok" if max_len_ok else "response body exceeds 600 characters",
        )
    )

    body_lower = draft.body.lower()
    banned_phrase = next((p for p in BANNED_RESPONSE_PHRASES if p in body_lower), None)
    banned_ok = banned_phrase is None
    events.append(
        _event(
            stage="draft",
            name="no_forbidden_claims",
            passed=banned_ok,
            reason="ok" if banned_ok else f"contains forbidden phrase: {banned_phrase}",
        )
    )

    next_step_markers = {
        "next",
        "follow",
        "we will",
        "i will",
        "our team will",
        "you can expect",
    }
    has_next_step = any(marker in body_lower for marker in next_step_markers)
    events.append(
        _event(
            stage="draft",
            name="contains_next_step",
            passed=has_next_step,
            reason="ok"
            if has_next_step
            else "response should include a next-step statement",
        )
    )

    return events


def _validate_final_compliance(draft: DraftResponse) -> list[GuardrailEvent]:
    events: list[GuardrailEvent] = []

    forbidden_tone_words = {"idiot", "stupid", "nonsense"}
    lower = draft.body.lower()
    tone_word = next((w for w in forbidden_tone_words if w in lower), None)
    tone_ok = tone_word is None
    events.append(
        _event(
            stage="compliance",
            name="professional_tone",
            passed=tone_ok,
            reason="ok" if tone_ok else f"contains unprofessional word: {tone_word}",
        )
    )

    return events


def _all_passed(events: list[GuardrailEvent]) -> bool:
    return all(event.passed for event in events)


def _print_audit(audit_trail: list[GuardrailEvent]) -> None:
    print("\nAudit trail:")
    for event in audit_trail:
        status = "PASS" if event.passed else "FAIL"
        print(f"  - [{event.stage}] {event.name}: {status} ({event.reason})")


def run_guardrailed_support_pipeline(ticket: SupportTicket) -> PipelineResult:
    """Run a support pipeline with guardrails at every stage."""
    print("=== Guardrailed Support Pipeline ===")
    print(
        f"Ticket {ticket.ticket_id} | tier={ticket.customer_tier} | urgency={ticket.urgency}"
    )

    audit_trail: list[GuardrailEvent] = []

    print("\n[ Stage 1 / 4 ] Intake validation...")
    intake_events = _validate_intake(ticket)
    audit_trail.extend(intake_events)
    if not _all_passed(intake_events):
        _print_audit(audit_trail)
        return PipelineResult(
            status="rejected",
            ticket_id=ticket.ticket_id,
            escalation_required=False,
            audit_trail=audit_trail,
        )

    print("[ Stage 2 / 4 ] Classification + routing...")
    classify_prompt = (
        "Classify this customer support ticket for routing.\n\n"
        f"ticket_id: {ticket.ticket_id}\n"
        f"customer_tier: {ticket.customer_tier}\n"
        f"urgency: {ticket.urgency}\n"
        f"message: {ticket.message}\n"
    )
    classification = classification_agent.run_sync(classify_prompt).output
    classification = _apply_local_confidence_adjustments(ticket, classification)

    classification_events = _validate_classification(classification)
    audit_trail.extend(classification_events)

    category_ok = next(
        e for e in classification_events if e.name == "category_allowed"
    ).passed
    confidence_ok = next(
        e for e in classification_events if e.name == "confidence_threshold"
    ).passed

    if not category_ok:
        _print_audit(audit_trail)
        return PipelineResult(
            status="rejected",
            ticket_id=ticket.ticket_id,
            escalation_required=False,
            classification=classification,
            audit_trail=audit_trail,
        )

    if not confidence_ok:
        _print_audit(audit_trail)
        return PipelineResult(
            status="escalated",
            ticket_id=ticket.ticket_id,
            escalation_required=True,
            classification=classification,
            audit_trail=audit_trail,
        )

    print("[ Stage 3 / 4 ] Drafting response...")
    draft_prompt = (
        "Write a support response.\n\n"
        f"Category: {classification.category}\n"
        f"Severity: {classification.severity}\n"
        f"Ticket message: {ticket.message}\n"
    )
    draft = draft_agent.run_sync(draft_prompt).output

    draft_events = _validate_draft(draft)
    audit_trail.extend(draft_events)
    if not _all_passed(draft_events):
        _print_audit(audit_trail)
        return PipelineResult(
            status="rejected",
            ticket_id=ticket.ticket_id,
            escalation_required=False,
            classification=classification,
            draft_response=draft,
            audit_trail=audit_trail,
        )

    print("[ Stage 4 / 4 ] Final compliance checks...")
    compliance_events = _validate_final_compliance(draft)
    audit_trail.extend(compliance_events)
    if not _all_passed(compliance_events):
        _print_audit(audit_trail)
        return PipelineResult(
            status="rejected",
            ticket_id=ticket.ticket_id,
            escalation_required=False,
            classification=classification,
            draft_response=draft,
            audit_trail=audit_trail,
        )

    _print_audit(audit_trail)
    return PipelineResult(
        status="approved",
        ticket_id=ticket.ticket_id,
        escalation_required=False,
        classification=classification,
        draft_response=draft,
        audit_trail=audit_trail,
    )


def _print_result(result: PipelineResult) -> None:
    print("\n" + "=" * 40)
    print(f"FINAL STATUS: {result.status.upper()}")
    print("=" * 40)

    if result.classification is not None:
        c = result.classification
        print(
            f"Classification: category={c.category}, severity={c.severity}, "
            f"confidence={c.confidence:.2f}"
        )

    if result.draft_response is not None:
        print(f"\nSubject: {result.draft_response.subject}")
        print(f"Body: {result.draft_response.body}")

    if result.escalation_required:
        print("\nAction: Route to human support queue.")


def main() -> None:
    demo_tickets = [
        SupportTicket(
            ticket_id="T-1001",
            customer_tier="pro",
            urgency="high",
            message=(
                "After today's update, the mobile app crashes every time I try to "
                "upload coursework. Please help me restore access quickly."
            ),
        ),
        SupportTicket(
            ticket_id="T-1002",
            customer_tier="free",
            urgency="medium",
            message="Not sure, something is wrong.",
        ),
        SupportTicket(
            ticket_id="T-1003",
            customer_tier="enterprise",
            urgency="urgent",
            message="Need billing support as soon as possible.",
        ),
    ]

    print("\n" + "#" * 60)
    result = run_guardrailed_support_pipeline(demo_tickets[0])
    _print_result(result)


if __name__ == "__main__":
    main()

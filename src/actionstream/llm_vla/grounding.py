"""Version 2 text-backed authorization, separate from the preserved v1 parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .contracts import CAPABILITY, TaskSpec, strict_object


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_hash(contract: dict) -> str:
    return digest_text(json.dumps(contract, sort_keys=True, ensure_ascii=False))


@dataclass(frozen=True)
class OriginalRequest:
    request_id: str
    text: str

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("A nonempty opaque request ID is required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Original user text is required")

    @property
    def text_sha256(self):
        return digest_text(self.text)


@dataclass(frozen=True)
class Mention:
    quote: str
    start: int
    end: int
    entity: str


@dataclass(frozen=True)
class GroundedTask:
    original: OriginalRequest
    original_sha256: str
    raw_model_output: str
    model_output_sha256: str
    contract_sha256: str
    target: Mention
    destination: Mention
    version: str = "2.0"

    def control_spec(self):
        """Host registry assembly, never a generated canonical instruction."""
        return TaskSpec(
            "1.0",
            self.original.request_id,
            "accept",
            **CAPABILITY,
            reason="v2 original text and exact entity mentions verified",
        )


def parse_decision(raw: str, contract: dict) -> dict:
    # Read the discriminator strictly before applying the corresponding union arm.
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate decision field")
            result[key] = value
        return result

    data = json.loads(raw, object_pairs_hook=reject_duplicates)
    if type(data) is not dict:
        raise ValueError("Decision must be an object")
    decision = data.get("decision")
    if decision == "accept":
        data = strict_object(raw, {"decision", "target_quote", "destination_quote"})
        if any(
            not isinstance(data[key], str) or not data[key].strip()
            for key in ("target_quote", "destination_quote")
        ):
            raise ValueError("Accepted intent requires two nonempty verbatim quotes")
    elif decision in ("reject", "unknown"):
        data = strict_object(raw, {"decision", "reason"})
        allowed = contract["reasons"][decision]
        if data["reason"] not in allowed:
            raise ValueError("Unknown reason code")
    else:
        raise ValueError("Invalid semantic decision")
    return data


def normalize_alias(quote: str) -> str:
    value = " ".join(quote.casefold().split())
    return re.sub(r"^(?:the|a|an) ", "", value)


def resolve_mention(request: OriginalRequest, quote: str, slot: str, contract: dict):
    matches = list(re.finditer(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", request.text))
    if len(matches) != 1:
        raise ValueError("reference_missing_or_nonunique")
    if normalize_alias(quote) not in contract["aliases"][slot]:
        raise ValueError("unsupported_or_unresolved_reference")
    match = matches[0]
    return Mention(quote, match.start(), match.end(), CAPABILITY[slot])


def adjudicate(request: OriginalRequest, raw: str, contract: dict) -> dict:
    """Exact mentions prove textual references, not complete semantic entailment."""
    result = dict(
        schema_status="VALID",
        raw_decision=None,
        decision="explicit_failure",
        reason=None,
        task=None,
    )
    try:
        data = parse_decision(raw, contract)
    except (ValueError, TypeError) as exc:
        return dict(
            result, schema_status="EXPLICIT_FAILURE", reason="malformed", error=str(exc)
        )
    result["raw_decision"] = data["decision"]
    if data["decision"] != "accept":
        return dict(result, decision=data["decision"], reason=data["reason"])
    try:
        target = resolve_mention(request, data["target_quote"], "target", contract)
        destination = resolve_mention(
            request, data["destination_quote"], "destination", contract
        )
    except ValueError as exc:
        return dict(result, reason=str(exc))
    # Conservative finite-scope guards are declared before evaluation. They do
    # not parse argument roles, additions or contradictory discourse generally.
    for guard in contract["text_guards"]:
        if re.search(guard["pattern"], request.text, flags=re.IGNORECASE):
            return dict(result, decision="reject", reason=guard["reason"])
    task = GroundedTask(
        request,
        request.text_sha256,
        raw,
        digest_text(raw),
        contract_hash(contract),
        target,
        destination,
    )
    return dict(
        result,
        decision="accept",
        reason="supported_with_textual_evidence",
        task=asdict(task),
    )


def materialize(request: OriginalRequest, raw: str, contract: dict) -> GroundedTask:
    verdict = adjudicate(request, raw, contract)
    if verdict["decision"] != "accept":
        raise ValueError(
            f"No executable authorization: {verdict['decision']}/{verdict['reason']}"
        )
    row = verdict["task"]
    return GroundedTask(
        request,
        row["original_sha256"],
        raw,
        row["model_output_sha256"],
        row["contract_sha256"],
        Mention(**row["target"]),
        Mention(**row["destination"]),
    )


def validate_authorization(
    request: OriginalRequest, task: GroundedTask, contract: dict
):
    if not isinstance(task, GroundedTask) or task.original != request:
        raise ValueError("Authorization belongs to another original request")
    expected = materialize(request, task.raw_model_output, contract)
    if expected != task:
        raise ValueError("Forged, changed or stale textual authorization")


def execute_authorized(request, task, contract, port, checker, emit, **kwargs):
    from .integration import execute

    validate_authorization(request, task, contract)
    emit("text_authorization", original=asdict(request), authorization=asdict(task))
    return execute(task.control_spec(), port, checker, emit, **kwargs)

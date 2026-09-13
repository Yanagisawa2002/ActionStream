"""Finite, whole-utterance authorization. No evaluator IDs or answer tables."""

from dataclasses import asdict, dataclass
import re

from .grounding import OriginalRequest, adjudicate, contract_hash, materialize

TOKEN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|[0-9]+|[^\w\s]", re.UNICODE)
# Every production describes one goal. Carry/lift/place are allowed ordered
# substeps on the same referent; no general conjunction erasure is performed.
ART = r"(?:(?:the|a|one|a single|the single) )?"
T, B = ART + "T", ART + "B"
V = r"(?:put|place|move|transfer|deposit|relocate|set|set down|get)"
P = r"(?:in|into|inside|to)"
PRE = r"(?:(?:please|could you|can you|would you|please could you|please can you|your task is to|all i need is for you to) )?"
POST = r"(?: (?:please|for me|as your only task))?"
SOURCE = r"(?: from where it is)?"
GRAMMAR = (
    rf"{V} {T}{SOURCE}(?: away)? {P} {B}",
    rf"(?:pick up|lift|take) {T} and (?:put|place|leave|set|set down|lower) it (?:down )?{P} {B}",
    rf"(?:pick up|lift|take) {T} carry it over and (?:put|place|set) it down {P} {B}",
    rf"(?:pick up|lift|take) {T} and carry it over and (?:put|place|set) it down {P} {B}",
    rf"carry {T} over to {B} and (?:place|put|leave) it inside",
    rf"{P} {B} {V} {T}",
    rf"{B} is where i want you to {V} {T}",
    rf"(?:i need|i would like) {T} (?:placed|put|moved|deposited) {P} {B}",
    rf"have {T} (?:moved|placed|put) {P} {B}",
    rf"finish with {T} resting {P} {B}",
    rf"{T} should go {P} {B}",
    rf"use {B} as the destination for {T}",
    rf"before you finish make sure you have placed {T} {P} {B}",
)
SCHEMA = dict(
    version="3.0",
    productions=GRAMMAR,
    prefix=PRE,
    suffix=POST,
    source="current location only; explicit counter is an unchecked constraint",
    quantities="one only",
    residual="unknown prevents authorization",
    ordering="lift/carry/place of the same explicitly bound target only",
)
ACTIONS = set(
    "put place move transfer deposit relocate set get pick lift take lower leave carry placed moved deposited resting go use finish close open bring wash rotate pour shake".split()
)
NEGATION = set("no not never don't cannot can't won't without".split())
QUANTITIES = set("a an one single two three both all another additional".split())
ORDER = set("and then before after first finally while".split())
CONSTRAINT = set(
    "if unless until carefully slowly upright gently care spilling only".split()
)


def tokens(text):
    return [(m.group().casefold(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def inspect_coverage(request: OriginalRequest, contract: dict) -> dict:
    raw = tokens(request.text)
    # Non-ASCII word characters omitted by the lexical pattern must never vanish.
    gaps = request.text
    for _, start, end in reversed(raw):
        gaps = gaps[:start] + " " * (end - start) + gaps[end:]
    aliases = []
    for slot, marker in (("target", "T"), ("destination", "B")):
        for alias in contract["aliases"][slot]:
            aliases.append(([x[0] for x in tokens(alias)], marker, slot))
    aliases.sort(key=lambda x: -len(x[0]))
    symbols, ledger = [], []
    index = 0
    while index < len(raw):
        found = next(
            (
                a
                for a in aliases
                if [x[0] for x in raw[index : index + len(a[0])]] == a[0]
            ),
            None,
        )
        size = len(found[0]) if found else 1
        word, start, _ = raw[index]
        end = raw[index + size - 1][2]
        symbol = found[1] if found else word
        kind = (
            found[2]
            if found
            else (
                "action"
                if word in ACTIONS
                else "negation"
                if word in NEGATION
                else "quantity"
                if word in QUANTITIES
                else "ordering"
                if word in ORDER
                else "constraint"
                if word in CONSTRAINT
                else "grammar_or_unresolved"
            )
        )
        ledger.append(
            dict(
                start=start,
                end=end,
                quote=request.text[start:end],
                kind=kind,
                symbol=symbol,
                disposition="unknown",
            )
        )
        symbols.append(symbol)
        index += size
    # These punctuation productions are explicit and appear in the ledger.
    normalized = " ".join(s for s in symbols if s != ",")
    normalized = normalized.replace(" ; ", " and ")
    normalized = re.sub(r" [.?]$", "", normalized)
    normalized = normalized.replace('item labeled " T "', "T")
    production = next(
        (
            i
            for i, rule in enumerate(GRAMMAR)
            if re.fullmatch(PRE + rule + POST, normalized)
        ),
        None,
    )
    forbidden = [x for x in ledger if x["kind"] == "negation"]
    valid = production is not None and not gaps.strip() and not forbidden
    # A partial parse never certifies an object, action or role from word presence.
    for row in ledger:
        row["disposition"] = (
            "supported" if valid else ("rejected" if row in forbidden else "unknown")
        )
    obligations = list(ledger)
    if valid:
        obligations += [
            dict(kind=k, value=v, disposition="supported")
            for k, v in (
                (
                    "role_binding",
                    {
                        "moved_object": "tomato_sauce",
                        "stationary_destination": "basket",
                    },
                ),
                ("quantity", 1),
                ("polarity", "affirmative"),
                (
                    "execution",
                    "one pick_place goal with optional ordered same-target substeps",
                ),
                ("coverage", "all source tokens consumed by finite production"),
            )
        ]
    else:
        obligations.append(
            dict(
                kind="unresolved_utterance",
                quote=request.text,
                disposition="unknown",
                reason="no complete supported derivation",
            )
        )
    return dict(
        complete=valid,
        original_sha256=request.text_sha256,
        grammar_sha256=contract_hash(SCHEMA),
        production=production,
        normalized=normalized,
        obligations=obligations,
        decision="accept" if valid else "reject" if forbidden else "unknown",
    )


def adjudicate_covered(request, raw, contract):
    old = adjudicate(request, raw, contract)
    coverage = inspect_coverage(request, contract)
    result = dict(old, coverage=coverage)
    if old["decision"] == "accept" and not coverage["complete"]:
        result.update(
            decision=coverage["decision"],
            reason="incomplete_semantic_coverage",
            task=None,
        )
    return result


@dataclass(frozen=True)
class CoveragePermit:
    original: OriginalRequest
    raw: str
    proof_sha256: str
    grammar_sha256: str
    contract_sha256: str


def authorize(request, raw, contract):
    result = adjudicate_covered(request, raw, contract)
    if result["decision"] != "accept":
        raise ValueError("No complete supported authorization")
    return CoveragePermit(
        request,
        raw,
        contract_hash(result),
        contract_hash(SCHEMA),
        contract_hash(contract),
    )


def execute_covered(request, permit, contract, port_factory, checker, emit, **kwargs):
    """New boundary re-derives proof BEFORE constructing any execution backend.

    The older research entrypoints remain separate. This is an application
    authorization boundary, not a Python sandbox against callers importing v1.
    DISPATCH_005's supervisor cannot run any native phase at all.
    """
    from .integration import execute

    if type(permit) is not CoveragePermit or permit.original != request:
        raise ValueError("Missing or mismatched coverage permit")
    if authorize(request, permit.raw, contract) != permit:
        raise ValueError("Changed, forged or stale coverage permit")
    task = materialize(request, permit.raw, contract)
    emit("complete_text_authorization", authorization=asdict(permit))
    return execute(task.control_spec(), port_factory(), checker, emit, **kwargs)

"""Prompt sourcing for the causal-refusal experiment.

Three sources, in order of preference for this experiment:

1. ``jbb``    -- JailbreakBench/JBB-Behaviors. 100 harmful + 100 benign, MIT.
                 The two splits are *index-matched*: pair i shares a Behavior,
                 a Category and a topic, differing only in harmfulness. That
                 matching is the whole point -- it means
                 mean(harmful) - mean(benign) isolates refusal rather than
                 topic, sentence form, or distributional shift.

2. ``offline``-- A 10-pair stratified subset of (1) vendored into
                 data/pilot_prompts.json, one pair per category. Runs with no
                 network and no token. This is the default for the smoke test.

3. ``advbench``-- AdvBench harmful (500, MIT) paired with Alpaca benign. This is
                 what "Refusal in Language Models Is Mediated by a Single
                 Direction" actually used, so it matters for reproduction --
                 but the pairing is *arbitrary*, not matched. Any direction it
                 yields may encode "sensitive topic" or distribution shift
                 alongside refusal. Use it as a second condition to check that
                 a finding is not an artifact of one dataset, not as the
                 primary source.

Gated repos (AdvBench on HF) need a token: `hf auth login`, or set HF_TOKEN.
AdvBench also has an ungated GitHub mirror, used as automatic fallback.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Repo root, regardless of whether we are imported from notebooks/ or src/.
_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_PATH = _ROOT / "data" / "pilot_prompts.json"

ADVBENCH_GITHUB = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


@dataclass
class PromptPair:
    """One harmful prompt and its benign counterpart.

    ``matched`` records whether the two are genuinely paired (JBB) or merely
    zipped together from unrelated pools (AdvBench+Alpaca). Carried through so
    downstream analysis cannot silently forget which regime produced a
    direction -- the interpretation of the result differs.
    """

    harmful: str
    benign: str
    category: str = "unknown"
    behavior: str = "unknown"
    index: int = -1
    matched: bool = True


def _load_offline(n: int | None) -> list[PromptPair]:
    if not OFFLINE_PATH.exists():
        raise FileNotFoundError(
            f"{OFFLINE_PATH} is missing. Re-fetch it with "
            "load_pilot_pairs(source='jbb') or restore it from version control."
        )
    blob = json.loads(OFFLINE_PATH.read_text())
    pairs = [
        PromptPair(
            harmful=p["harmful"],
            benign=p["benign"],
            category=p.get("category", "unknown"),
            behavior=p.get("behavior", "unknown"),
            index=p.get("index", -1),
            matched=True,
        )
        for p in blob["pairs"]
    ]
    return pairs if n is None else pairs[:n]


def _load_jbb(n: int | None) -> list[PromptPair]:
    """Load JBB from the Hub, stratified across categories.

    Stratifying matters even for a smoke test: drawing the first n rows would
    take every prompt from a single category, and a direction computed from one
    theme is a topic direction wearing a refusal costume.
    """
    from datasets import load_dataset

    harmful = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    benign = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")

    by_index = {row["Index"]: row for row in benign}
    pairs: list[PromptPair] = []
    for row in harmful:
        mate = by_index.get(row["Index"])
        if mate is None:  # defensive: upstream split drift
            continue
        pairs.append(
            PromptPair(
                harmful=row["Goal"],
                benign=mate["Goal"],
                category=row["Category"],
                behavior=row["Behavior"],
                index=row["Index"],
                matched=True,
            )
        )

    if n is None:
        return pairs

    # Round-robin across categories so a truncated sample stays balanced.
    buckets: dict[str, list[PromptPair]] = {}
    for p in pairs:
        buckets.setdefault(p.category, []).append(p)
    out: list[PromptPair] = []
    while len(out) < n and any(buckets.values()):
        for cat in sorted(buckets):
            if buckets[cat] and len(out) < n:
                out.append(buckets[cat].pop(0))
    return out


def _fetch_advbench_github() -> list[str]:
    with urllib.request.urlopen(ADVBENCH_GITHUB, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    return [row["goal"] for row in csv.DictReader(io.StringIO(text))]


def _load_advbench(n: int | None) -> list[PromptPair]:
    """AdvBench harmful + Alpaca benign -- the paper's configuration.

    The HF copy of AdvBench is gated; fall back to the ungated GitHub CSV from
    the original llm-attacks repo so this works with or without a token.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("walledai/AdvBench", split="train")
        harmful = [r["prompt"] for r in ds]
    except Exception as exc:  # gated, no token, or offline
        print(f"  HF AdvBench unavailable ({type(exc).__name__}); using GitHub mirror.")
        harmful = _fetch_advbench_github()

    # Alpaca entries with a non-empty `input` are context-dependent and read as
    # fragments without it; keep only self-contained instructions.
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign = [r["instruction"] for r in alpaca if not r["input"].strip()]

    count = min(len(harmful), len(benign)) if n is None else n
    return [
        PromptPair(
            harmful=harmful[i],
            benign=benign[i],
            category="advbench+alpaca",
            behavior="unknown",
            index=i,
            matched=False,  # arbitrary pairing -- see module docstring
        )
        for i in range(count)
    ]


def load_pilot_pairs(source: str = "offline", n: int | None = 8) -> list[PromptPair]:
    """Return harmful/benign prompt pairs.

    Args:
        source: ``"offline"`` (vendored JBB subset, default), ``"jbb"``
            (full JBB from the Hub), or ``"advbench"`` (paper configuration).
        n: How many pairs; ``None`` for all available.
    """
    loaders = {"offline": _load_offline, "jbb": _load_jbb, "advbench": _load_advbench}
    if source not in loaders:
        raise ValueError(f"source must be one of {sorted(loaders)}, got {source!r}")
    pairs = loaders[source](n)
    if not pairs:
        raise RuntimeError(f"source {source!r} returned no pairs")
    return pairs


def describe(pairs: list[PromptPair]) -> str:
    """One-line summary for the notebook's provenance record."""
    cats = sorted({p.category for p in pairs})
    kind = "matched" if all(p.matched for p in pairs) else "UNMATCHED"
    return f"{len(pairs)} pairs ({kind}) across {len(cats)} categor(y/ies): {', '.join(cats)}"

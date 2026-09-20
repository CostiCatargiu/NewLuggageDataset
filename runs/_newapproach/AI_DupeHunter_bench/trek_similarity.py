"""
TREK SYT vs SWT Duplicate Detection
====================================
Hybrid BM25 + embedding similarity scoring to flag SYT (system) test cases
that are likely duplicates or near-duplicates of their linked SWT (software)
test cases -- i.e. cases where the SWT was copy-pasted from the SYT (or vice
versa) without meaningfully adapting the test for the SW level, indicating
redundant test coverage rather than a genuine system-to-software refinement.

Design rationale (adapted from the existing RAG pipeline in
C:\\DISK\\MAIA\\vehicleConvert\\xx_en_sys_ai_tc_generation\\src):

  - That codebase uses LangChain + Chroma + BM25 for full retrieval over
    large document catalogs (thousands of method-catalog entries). That
    retrieval infrastructure (vector DB persistence, ensemble retrievers,
    parent-document chunking) is unnecessary here: a traceability run only
    ever has a small, already-known set of SYT<->SWT pairs (dozens, not
    thousands), so this module does direct pairwise scoring instead of
    building a searchable index.

  - What IS reused, exactly as implemented there:
      * The embeddings client construction pattern (OpenAIEmbeddings
        pointed at the internal LLM gateway, same model/env vars/headers)
        -- see base_organization_retriever.py:83-132.
      * Exact-dedup pre-pass before spending any embedding call (mirrors
        TestStepClusterer._deduplicate()).
      * L2-normalize embeddings, then cosine similarity via dot product
        (mirrors TestStepClusterer._embed() / _merge_similar_clusters()
        and ClusterRAGOrchestrator._select_representatives()'s
        `sim = mat @ mat.T` pattern).
      * BM25 scoring normalized to [0,1] by dividing by the max raw score
        across the whole corpus (mirrors ANSRetriever._get_bm25_internals()
        / retrieve_with_scores()'s "compute scores for ALL documents, not
        just top-k, then normalize by max" approach) -- rank_bm25 is used
        directly here (same engine LangChain's BM25Retriever wraps).
      * A straight weighted-average fusion formula:
        `ensemble = bm25_weight*bm25 + vec_weight*vector`. NOTE: this
        module deliberately does NOT reuse retrieve_with_scores()'s
        "if BM25 < 0.3, discard it and use vector alone" fallback rule --
        that rule fits a retrieval/ranking context (many candidate
        documents, ranking still needs to work when one has zero lexical
        overlap) but is wrong for comparing two specific known texts: it
        would silently override whatever weight the user configured
        (e.g. 50/50) essentially every time, since full test-case text
        commonly has weak lexical overlap even between genuine
        duplicates. See ensemble_score()'s docstring for the full
        rationale.
      * Threshold-based classification bands, calibrated from the same
        cosine-distance thresholds used there for cluster merging
        (merge_threshold=0.15, rescue_threshold=0.25/0.35 distance ==
        0.85/0.75/0.65 similarity) as a starting point, now applied to the
        final ensemble score rather than pure cosine similarity.

  - What is DIFFERENT from the RAG codebase's BM25 usage: that codebase
    computes BM25 as *query vs. large document catalog* (many candidate
    documents, one query). Here we're comparing exactly two known texts
    (one SYT, one SWT), so BM25 needs a corpus to compute meaningful IDF
    weighting against. This module builds that corpus from ALL unique SYT
    and SWT texts in the current comparison batch (deduplicated), then for
    each pair scores syt-as-query-against-corpus at the SWT's corpus
    position, and vice versa, and averages the two directions (BM25 is
    inherently asymmetric; duplicate detection has no natural "query"
    side, so symmetric averaging is the more defensible choice here).

Credentials: the LLM gateway URL is HARDCODED (LLM_GATEWAY_URL, below) --
discovering duplicate SYT/SWT test cases is this application's core
purpose, not an optional add-on, so there is exactly one gateway to talk
to. Each user's JWT Token is a per-user credential, entered once in
ProjectSetupDialog and stored per-project (see trek_projects.py); it is
MANDATORY -- every project requires one, and `get_embeddings_client()`
raises a clear RuntimeError if it's missing rather than failing deep
inside a network call.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

EMBEDDING_MODEL = "google.text-multilingual-embedding-002"

# Per-input-token price ($/token) for embedding models, used as a fallback
# whenever the gateway's x-litellm-response-cost header comes back 0 for an
# embedding call (observed in practice: the gateway prices/reports chat
# completions but not this embedding model) -- from the LLM Incubator
# gateway's own /models pricing table ($0.1000 / 1M input tokens, output
# N/A since embeddings have no output tokens).
EMBEDDING_MODEL_PRICING = {
    "google.text-multilingual-embedding-002": 0.10 / 1_000_000,
}


# Chat model for the optional LLM-judge stage (see judge_pair_with_llm() /
# run_llm_judge_stage()) -- reads both test cases' actual text/logic to
# catch redundant coverage the hybrid BM25/vector/sequence score misses
# when SYT/SWT pairs are phrased very differently despite testing the same
# scenario (or vice versa: similarly-worded pairs that test different things).
CHAT_MODEL = "gpt-5-mini"

# Chat models the user can pick from for the LLM-judge stage (header ✎
# Edit Project's gateway is shared across all of them). Kept in sync with
# vehicleConvert's src/utils/token_stats.py MODEL_PRICING keys -- that's
# the authoritative list of models known to work against this same
# LLM Incubator gateway, so TREK's dropdown reuses it rather than
# maintaining a second, potentially-drifting list.
AVAILABLE_CHAT_MODELS = [
    # ── GPT (Azure) ────────────────────────────────────────────
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5.1",
    "gpt-5.1-chat",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-35-turbo",
    "gpt-o3-mini",
    # ── Claude (Vertex AI) ─────────────────────────────────────
    "vertex_ai/claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-5:reasoning",
    "claude-4-6-opus-v1:0",
    "claude-4-7-opus-v1:0",
    "claude-4-8-opus-v1:0",
    # ── Gemini (Vertex AI) ─────────────────────────────────────
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "failover/gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    # ── Llama ──────────────────────────────────────────────────
    "meta-llama/Llama-3.3-70B-Instruct",
]

# Hardcoded LLM gateway host (same internal gateway used by the
# vehicleConvert AI test-generation tooling). Not user-configurable --
# duplicate detection is this application's core purpose, so there is
# exactly one gateway to talk to. '/llm' is appended by
# get_embeddings_client() to match that codebase's URL construction.
LLM_GATEWAY_URL = "https://api.llm-incubator.automotive.cloud/dev/v0"

# Default hybrid weighting across THREE signals: BM25 (lexical), vector
# (semantic), and method-call sequence similarity (structural -- see
# extract_method_call_sequence() / method_sequence_similarity() below).
#
# BM25 kept low (0.15): duplicate-detection corpora are always small (just
# the current run's SYT/SWT texts) and genuinely duplicate/near-duplicate
# test cases share most of their vocabulary by definition. BM25's IDF
# weighting heavily down-weights any term appearing in >=50% of a small
# corpus -- verified directly (see BM25CorpusScorer tests): two
# near-identical sentences differing by one word scored only ~0.48 on
# BM25 alone, because nearly every shared word (most of the sentence) got
# IDF-clamped to near-zero.
#
# Sequence weighted meaningfully (0.30): TREK test procedures in this
# domain are scripted Component.method(...) call sequences, not free
# prose. Two tests calling the SAME methods in the SAME order is a much
# more literal, harder-to-fake "this was copy-pasted" signal than either
# BM25 or embedding similarity, which both cluster same-TOPIC text
# together even when the actual tested behavior differs (verified
# empirically: a "reset persistence" test and a "session-transition state
# tracking" test in the same flash-mode feature scored vector=0.86 despite
# testing different things -- their method sequences are completely
# different and would correctly show that). When a test case has no
# scripted procedure at all (pure prose/manual test), this dimension is
# excluded from the ensemble entirely rather than forced to 0 (see
# ensemble_score()) so it never unfairly penalizes non-scripted pairs.
#
# Vector remains the largest single weight (0.55) as the most generally
# reliable semantic signal, but sequence and BM25 together now provide a
# meaningful counterbalance grounded in the actual literal test content
# rather than pure "same topic" clustering.
DEFAULT_BM25_WEIGHT = 0.15
DEFAULT_VEC_WEIGHT = 0.55
DEFAULT_SEQ_WEIGHT = 0.30

# Cosine-SIMILARITY / ensemble-score thresholds, calibrated starting from
# the RAG codebase's cosine-DISTANCE thresholds for cluster merging (0.15
# merge / 0.25-0.35 rescue). These are intentionally conservative starting
# points for a NEW comparison domain -- expect to retune after reviewing
# real results. Applied to the final ENSEMBLE score, not pure cosine
# similarity, since BM25 can now also contribute.
SIMILARITY_DUPLICATE      = 0.95   # near-identical -- almost certainly copy-pasted
SIMILARITY_NEAR_DUPLICATE = 0.85   # substantially similar -- likely redundant coverage
SIMILARITY_SIMILAR        = 0.70   # some meaningful overlap -- worth a human look

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Pattern to match component method calls: Component.method(...). Reused
# verbatim from test_script_review.py's METHOD_CALL_PATTERN in the
# vehicleConvert AI test-generation codebase, which already solved this
# exact extraction problem for the same kind of scripted test procedures
# (e.g. "SWUpdates.ApplyStateFlashModeActive()", "DiagInterface.CheckDataByID(...)").
_METHOD_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(")


def classify_similarity(
    score: float,
    duplicate_threshold: float = SIMILARITY_DUPLICATE,
    near_duplicate_threshold: float = SIMILARITY_NEAR_DUPLICATE,
    similar_threshold: float = SIMILARITY_SIMILAR,
) -> str:
    """Map an ensemble similarity score to a human label. Thresholds
    default to this module's calibrated constants but can be overridden
    per-call (e.g. from the DuplicateCheckSettingsDialog) so the user can
    tune sensitivity from the GUI without editing code."""
    if score >= duplicate_threshold:
        return "duplicate"
    if score >= near_duplicate_threshold:
        return "near_duplicate"
    if score >= similar_threshold:
        return "similar"
    return "distinct"


CLASSIFICATION_LABELS = {
    "duplicate":      ("🔴 Duplicate",      "#e74c3c"),
    "near_duplicate": ("🟠 Near-duplicate",  "#f5a623"),
    "similar":        ("🟡 Similar",         "#f5c518"),
    "distinct":        ("🟢 Distinct",        "#2ecc71"),
    "not_scored":      ("⚪ Not scored (RAG disabled)", "#888899"),
}

# Labels for the optional LLM-judge verdict (separate from the hybrid-score
# CLASSIFICATION_LABELS above -- this is the LLM's own read of the actual
# test logic, not a score threshold).
LLM_VERDICT_LABELS = {
    "same_scenario":      ("🧠🔴 LLM: Same scenario",      "#e74c3c"),
    "partial_overlap":    ("🧠🟡 LLM: Partial overlap",    "#f5c518"),
    "different_scenario": ("🧠🟢 LLM: Different scenario", "#2ecc71"),
    "error":              ("🧠⚠️ LLM: Judge error",         "#888899"),
}

# One-line plain-English meaning of each status above -- shown in the GUI
# (tooltips / detail panel) so a reviewer doesn't have to guess what e.g.
# "partial_overlap" actually implies about the pair.
CLASSIFICATION_EXPLANATIONS = {
    "duplicate": "Very high combined lexical/semantic/structural similarity -- almost certainly the same test restated.",
    "near_duplicate": "High similarity with minor differences -- likely redundant, worth a quick manual look.",
    "similar": "Meaningful overlap, but with enough difference that this may be legitimate distinct coverage.",
    "distinct": "Low similarity -- the hybrid BM25/vector/sequence score considers these two different tests.",
    "not_scored": "RAG scoring was disabled for this run (LLM-only verification, if enabled).",
}

LLM_VERDICT_EXPLANATIONS = {
    "same_scenario": "The counterpart verifies the IDENTICAL trigger and expected result as the SYT -- essentially redundant coverage.",
    "partial_overlap": "Shares the same general trigger/requirement area, but differs meaningfully in scope, depth, or verification mechanism (e.g. an extra check, or the same behavior verified a different way).",
    "different_scenario": "Exercises a genuinely different trigger, fault, boundary value, mode, or requirement aspect -- distinct coverage, not a duplicate.",
    "error": "The LLM call failed or returned an unparseable/missing response for this pair -- not a real verdict, treat as unjudged.",
}


def _normalize_for_exact_match(text: str) -> str:
    """Fold whitespace/case differences so a pure copy-paste duplicate is
    caught for free before spending an embedding call (mirrors
    TestStepClusterer._deduplicate()'s normalization)."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word/token extraction for BM25. Deliberately
    permissive (keeps underscores) so DFR tokens and method-style
    identifiers (e.g. 'o_SEV_Latent_Status') tokenize as meaningful whole
    units rather than being split apart."""
    return _TOKEN_RE.findall(text.lower())


def build_comparison_text(tc: dict, include_postcondition: bool = False) -> str:
    """Build the text used for similarity comparison from a TC content
    dict (same shape as returned by TrekExportLinksAPI.get_test_case_content
    and rendered by trek_gui._tc_to_html): Name + PreCondition + Procedure +
    Expected Result.

    Postcondition is EXCLUDED by default for real TREK data (used by
    DuplicateCheckWorker) -- cleanup steps are rarely diagnostic of
    whether two test cases test the same thing, and including them adds
    noise that dilutes the signal from the actually-meaningful fields.

    include_postcondition=True is used by the manual "Test Algorithm"
    sandbox (TestAlgorithmDialog), whose input fields are
    Name/PreCondition/Procedure/Postcondition (no Expected_result field at
    all) -- when set, Postcondition is included instead of/in addition to
    Expected_result so hand-typed synthetic data with only a Postcondition
    field still produces meaningful comparison text.
    """
    if not tc:
        return ""
    parts = [
        tc.get("Name", ""),
        tc.get("PreCondition", ""),
        tc.get("Procedure", ""),
        tc.get("Expected_result", ""),
    ]
    if include_postcondition:
        parts.append(tc.get("Postcondition", ""))
    parts = [p for p in parts if p and p.strip() and p.strip().lower() != "not used"]
    return "\n".join(parts).strip()


def extract_method_call_sequence(tc: dict) -> List[str]:
    """Extract the ORDERED sequence of scripted method calls from a test
    case's Procedure (and PreCondition) fields, e.g.:

        SWUpdates.ApplyStateFlashModeActive()
        DiagInterface.CheckDataByID(parameter=..., value=...)
        Battery.SetPowerOnReset()

    becomes:
        ["SWUpdates.ApplyStateFlashModeActive", "DiagInterface.CheckDataByID",
         "Battery.SetPowerOnReset"]

    Why this matters: TREK test procedures in this domain are scripted
    method-call sequences, not free prose (see the SW Update flash-mode
    examples this was built to handle). Two tests can share almost
    identical vocabulary (BM25) and land in the same embedding-similarity
    neighborhood (same feature area) while testing completely different
    behavior -- e.g. one triggers via ECU reset, the other via session
    control, using entirely different method calls in a different order.
    Conversely, a genuine copy-paste duplicate will call the SAME methods
    in the SAME order, even if variable names/comments/values differ
    slightly. Comparing the call sequence directly (see
    method_sequence_similarity()) gives a much sharper, more literal
    "was this copy-pasted" signal than either BM25 or embeddings alone.

    Only Component.method(...) calls are extracted (matches
    test_script_review.py's METHOD_CALL_PATTERN from the vehicleConvert
    codebase) -- bare comments, step numbers ("#1. Perform a POR..."), and
    prose descriptions are ignored, keeping the sequence to just the
    actual scripted actions.
    """
    if not tc:
        return []
    combined = "\n".join([
        tc.get("PreCondition", "") or "",
        tc.get("Procedure", "") or "",
    ])
    return _METHOD_CALL_PATTERN.findall(combined)


def method_sequence_similarity(seq_a: List[str], seq_b: List[str]) -> float:
    """Order-sensitive similarity between two method-call sequences, using
    difflib.SequenceMatcher's ratio (Ratcliff/Obershelp algorithm -- same
    family of technique used for diff tools) rather than a simple set/bag
    overlap, since call ORDER matters for detecting genuine copy-paste
    duplicates (same methods in a different order is a materially
    different test, e.g. "reset then check" vs. "check then reset").

    Returns 1.0 for identical sequences, 0.0 if either sequence is empty
    (no scripted procedure to compare -- e.g. a purely manual/prose test
    case), scaled proportionally in between based on the length and
    ordering of the longest common matching blocks.
    """
    if not seq_a or not seq_b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, seq_a, seq_b).ratio()


@dataclass(slots=True)
class SimilarityResult:
    syt_id: str
    swt_id: str
    syt_text: str = ""
    swt_text: str = ""
    syt_methods: List[str] = field(default_factory=list)
    swt_methods: List[str] = field(default_factory=list)
    score: float = 0.0            # final ensemble score (what classification is based on)
    bm25_score: float = 0.0       # normalized [0,1] lexical overlap score
    vector_score: float = 0.0     # cosine similarity of embeddings
    sequence_score: float = 0.0   # order-sensitive method-call sequence similarity
    has_sequence: bool = False    # False = sequence_score is N/A (0.0 is a placeholder, not a real score)
    classification: str = "distinct"
    exact_match: bool = False
    llm_verdict: Optional[str] = None   # None = LLM-judge stage not run for this pair
    llm_reasoning: str = ""
    counterpart_type: str = "SWT"   # "SWT" or "related_syt" -- what swt_id/swt_text actually is
    syt_tc: dict = field(default_factory=dict)  # original structured content (Name/PreCondition/Procedure/...)
    swt_tc: dict = field(default_factory=dict)  # -- lets the GUI render labeled sections instead of flat text;
                                                 # empty for pairs from older cached runs (display falls back to syt_text/swt_text)
    llm_cost_usd: float = 0.0       # REAL cost of judging THIS pair (cache hit or fresh) -- lets a
                                     # merged/superset result sum true historical LLM cost per pair,
                                     # with no double-counting (unlike embedding cost, which is shared
                                     # across pairs and NOT tracked per-pair for that reason).
    checked_at: str = ""            # ISO timestamp of the run that last (re)computed this pair --
                                     # lets the GUI flag which pairs are from the LATEST run in a
                                     # merged/superset cached result (see merge_duplicate_check_results()).
    # ── Manual review fields (filled by a human reviewer in the GUI) ──
    review_status: str = ""          # "" = unreviewed, "confirmed" = agree with verdict,
                                     # "disputed" = disagree with verdict
    review_comment: str = ""         # free-text comment from the reviewer (why they dispute, remarks, etc.)
    reviewer: str = ""               # who reviewed (Windows username auto-filled by the GUI)
    reviewed_at: str = ""            # ISO timestamp of the review


def _summary_counts(pairs: List["SimilarityResult"]) -> Dict[str, int]:
    counts = {"duplicate": 0, "near_duplicate": 0, "similar": 0, "distinct": 0, "not_scored": 0}
    for p in pairs:
        counts[p.classification] = counts.get(p.classification, 0) + 1
    return counts


def _llm_verdict_counts(pairs: List["SimilarityResult"]) -> Dict[str, int]:
    counts = {"same_scenario": 0, "partial_overlap": 0, "different_scenario": 0, "error": 0}
    for p in pairs:
        if p.llm_verdict:
            counts[p.llm_verdict] = counts.get(p.llm_verdict, 0) + 1
    return counts


def _agreement_counts(pairs: List["SimilarityResult"]) -> Dict[str, int]:
    counts = {"agree": 0, "disagree": 0, "not_applicable": 0}
    for p in pairs:
        rag_scored = p.exact_match or p.classification != "not_scored"
        if not rag_scored or not p.llm_verdict or p.llm_verdict == "error":
            counts["not_applicable"] += 1
            continue
        rag_flagged = p.exact_match or p.classification in ("duplicate", "near_duplicate", "similar")
        same_direction = (
            (rag_flagged and p.llm_verdict == "same_scenario")
            or (not rag_flagged and p.llm_verdict == "different_scenario")
        )
        counts["agree" if same_direction else "disagree"] += 1
    return counts


@dataclass
class DuplicateCheckResult:
    pairs: List[SimilarityResult] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)   # TC ids with no usable text
    total_tokens_embedded: int = 0   # EXACT tokens (from the gateway's usage.total_tokens) for NEW embeddings this run
    total_texts_embedded: int = 0    # count of distinct NEW texts embedded this run (cache hits excluded)
    cached_texts_embedded: int = 0   # count of distinct texts served from the embedding cache this run (no cost)
    total_cost_usd: float = 0.0      # cost for NEW embeddings this run: gateway-reported, falling back to EMBEDDING_MODEL_PRICING if the gateway reports 0
    historical_embed_cost_usd: float = 0.0  # REAL total cost of everything shown: this run's fresh spend + recorded cost of cache hits
    total_llm_tokens: int = 0        # EXACT tokens used by the optional LLM-judge stage (0 if not run)
    total_llm_cost_usd: float = 0.0  # gateway-reported cost for the LLM-judge stage (0.0 if not reported)
    historical_llm_cost_usd: float = 0.0  # REAL total cost of everything shown: this run's fresh spend + recorded cost of cache hits
    llm_calls: int = 0               # number of batch requests actually sent to the LLM stage
    llm_cached_pairs: int = 0        # pairs served from the LLM-judgment cache this run (no API call spent)
    rag_duration_seconds: float = 0.0   # wall-clock time spent in the RAG (BM25+vector+sequence) stage
    llm_duration_seconds: float = 0.0   # wall-clock time spent in the LLM-judge stage
    llm_model: str = ""               # chat model actually used for the LLM-judge stage (empty if not run)

    def by_classification(self, classification: str) -> List[SimilarityResult]:
        return [p for p in self.pairs if p.classification == classification]

    def summary_counts(self) -> Dict[str, int]:
        return _summary_counts(self.pairs)

    def llm_verdict_counts(self) -> Dict[str, int]:
        """Counts of pairs per LLM verdict (same_scenario/partial_overlap/
        different_scenario/error), for pairs where the LLM stage actually
        ran. Separate from summary_counts() -- the RAG classification and
        the LLM verdict are independent results, never merged."""
        return _llm_verdict_counts(self.pairs)

    def counterpart_type_counts(self) -> Dict[str, int]:
        """Counts of pairs by counterpart kind (SWT vs related_syt) -- so
        SYT-only chains (a SYR with a related SYT but no SWR/SWT bridge)
        are visibly accounted for in the metrics view, not just SYT-vs-SWT
        pairs."""
        counts: Dict[str, int] = {}
        for p in self.pairs:
            counts[p.counterpart_type] = counts.get(p.counterpart_type, 0) + 1
        return counts

    def agreement_counts(self) -> Dict[str, int]:
        """Counts of pairs where BOTH the RAG classification and the LLM
        verdict are available, split into 'agree' / 'disagree' -- plus
        'not_applicable' for pairs where RAG scoring was disabled, the LLM
        stage never ran for that pair, or the LLM judge errored. Mirrors
        the same agree/disagree logic used for the per-pair indicator in
        SideBySideResultsWidget._on_row_selected(): RAG counts as
        "flagged" for duplicate/near_duplicate/similar/exact_match, and
        they agree when both sides point the same direction (both flagged
        as redundant, or both consider the pair distinct).
        """
        return _agreement_counts(self.pairs)

    def compute_metrics(self) -> dict:
        """Aggregate everything the GUI's 'Metrics' view needs: per-method
        classification percentages, RAG/LLM agreement rate, pair counts by
        counterpart kind (SWT vs related_syt), and the REAL cost paid this
        run plus the REAL historical cost of anything served from cache
        (see historical_embed_cost_usd / historical_llm_cost_usd -- these
        are recorded per-item at the time each embedding/judgment was
        first computed, not an estimate).
        """
        total = len(self.pairs)
        rag_counts = self.summary_counts()
        rag_percentages = {k: (v / total * 100.0 if total else 0.0) for k, v in rag_counts.items()}

        llm_counts = self.llm_verdict_counts()
        llm_judged_total = sum(llm_counts.values())
        llm_percentages = {k: (v / llm_judged_total * 100.0 if llm_judged_total else 0.0) for k, v in llm_counts.items()}

        agreement = self.agreement_counts()
        agreement_applicable = agreement["agree"] + agreement["disagree"]
        agreement_pct = (agreement["agree"] / agreement_applicable * 100.0) if agreement_applicable else 0.0

        def _review_counts(pairs: List[SimilarityResult]) -> dict:
            counts = {"same_scenario": 0, "partial_overlap": 0, "different_scenario": 0, "unreviewed": 0}
            for p in pairs:
                if p.review_status in counts:
                    counts[p.review_status] += 1
                else:
                    counts["unreviewed"] += 1
            return counts

        def _llm_vs_human(pairs: List[SimilarityResult]) -> dict:
            """Compare LLM verdict to human review where both exist."""
            counts = {"agree": 0, "disagree": 0, "not_comparable": 0}
            for p in pairs:
                if not p.llm_verdict or p.llm_verdict == "error" or not p.review_status:
                    counts["not_comparable"] += 1
                elif p.llm_verdict == p.review_status:
                    counts["agree"] += 1
                else:
                    counts["disagree"] += 1
            return counts

        def _llm_accuracy_breakdown(pairs: List[SimilarityResult]) -> dict:
            """For each LLM verdict, how many pairs have been reviewed by a
            human, and of those how many agree/disagree + what the human said.

            Returns: {
                "same_scenario": {"total": 8, "reviewed": 3, "agree": 2, "disagree": 1, "not_reviewed": 5,
                                  "human_said": {"same_scenario": 2, "partial_overlap": 0, "different_scenario": 1}},
                "partial_overlap": {...},
                "different_scenario": {...},
            }
            """
            result = {}
            for llm_v in ("same_scenario", "partial_overlap", "different_scenario"):
                bucket = [p for p in pairs if p.llm_verdict == llm_v]
                reviewed = [p for p in bucket if p.review_status]
                human_said = {"same_scenario": 0, "partial_overlap": 0, "different_scenario": 0}
                agree = 0
                for p in reviewed:
                    if p.review_status in human_said:
                        human_said[p.review_status] += 1
                    if p.review_status == llm_v:
                        agree += 1
                result[llm_v] = {
                    "total": len(bucket),
                    "reviewed": len(reviewed),
                    "agree": agree,
                    "disagree": len(reviewed) - agree,
                    "not_reviewed": len(bucket) - len(reviewed),
                    "human_said": human_said,
                }
            return result

        def _breakdown_for(pairs: List[SimilarityResult]) -> dict:
            """Same rag_counts/llm_counts/agreement_counts/percentages shape
            as the top-level metrics, scoped to one counterpart_type -- lets
            the GUI show 'SYT vs SWT' and 'SYT vs SYT (related)' duplicates
            separately instead of one merged bucket."""
            sub_total = len(pairs)
            sub_rag_counts = _summary_counts(pairs)
            sub_llm_counts = _llm_verdict_counts(pairs)
            sub_llm_total = sum(sub_llm_counts.values())
            sub_agreement = _agreement_counts(pairs)
            sub_agreement_applicable = sub_agreement["agree"] + sub_agreement["disagree"]
            return {
                "total_pairs": sub_total,
                "rag_counts": sub_rag_counts,
                "rag_percentages": {k: (v / sub_total * 100.0 if sub_total else 0.0) for k, v in sub_rag_counts.items()},
                "llm_counts": sub_llm_counts,
                "llm_percentages": {k: (v / sub_llm_total * 100.0 if sub_llm_total else 0.0) for k, v in sub_llm_counts.items()},
                "llm_judged_total": sub_llm_total,
                "agreement_counts": sub_agreement,
                "agreement_percentage": (sub_agreement["agree"] / sub_agreement_applicable * 100.0) if sub_agreement_applicable else 0.0,
                "review_counts": _review_counts(pairs),
                "llm_vs_human": _llm_vs_human(pairs),
                "llm_accuracy_breakdown": _llm_accuracy_breakdown(pairs),
            }

        breakdown_by_counterpart_type = {
            "SWT": _breakdown_for([p for p in self.pairs if p.counterpart_type == "SWT"]),
            "related_syt": _breakdown_for([p for p in self.pairs if p.counterpart_type == "related_syt"]),
        }

        # Historical cost falls back to this run's fresh spend if the
        # caller never wired historical tracking (e.g. the manual Test
        # Algorithm sandbox doesn't use the persistent cache at all).
        embed_historical = max(self.historical_embed_cost_usd, self.total_cost_usd)
        llm_historical = max(self.historical_llm_cost_usd, self.total_llm_cost_usd)
        embed_cost_saved = embed_historical - self.total_cost_usd
        llm_cost_saved = llm_historical - self.total_llm_cost_usd

        return {
            "total_pairs": total,
            "counterpart_type_counts": self.counterpart_type_counts(),
            "rag_counts": rag_counts,
            "rag_percentages": rag_percentages,
            "llm_counts": llm_counts,
            "llm_percentages": llm_percentages,
            "llm_judged_total": llm_judged_total,
            "agreement_counts": agreement,
            "agreement_percentage": agreement_pct,
            "breakdown_by_counterpart_type": breakdown_by_counterpart_type,
            "embed_cost_usd": self.total_cost_usd,
            "embed_cached_texts": self.cached_texts_embedded,
            "embed_cost_historical_usd": embed_historical,
            "embed_cost_saved_usd": embed_cost_saved,
            "embed_tokens": self.total_tokens_embedded,
            "llm_cost_usd": self.total_llm_cost_usd,
            "llm_cached_pairs": self.llm_cached_pairs,
            "llm_cost_historical_usd": llm_historical,
            "llm_cost_saved_usd": llm_cost_saved,
            "llm_tokens": self.total_llm_tokens,
            "rag_duration_seconds": self.rag_duration_seconds,
            "llm_duration_seconds": self.llm_duration_seconds,
            "llm_model": self.llm_model,
            "review_counts": _review_counts(self.pairs),
            "llm_vs_human": _llm_vs_human(self.pairs),
            "llm_accuracy_breakdown": _llm_accuracy_breakdown(self.pairs),
        }

    def to_dict(self) -> dict:
        """Serialize this result for JSON export/reporting -- per-pair detail
        plus the aggregate token/cost totals for both the embedding (RAG)
        stage and the optional LLM-judge stage, so cost can be tracked and
        reported on later without re-running anything."""
        return {
            "summary_counts": self.summary_counts(),
            "llm_verdict_counts": self.llm_verdict_counts(),
            "pairs": [asdict(p) for p in self.pairs],
            "skipped": self.skipped,
            "total_tokens_embedded": self.total_tokens_embedded,
            "total_texts_embedded": self.total_texts_embedded,
            "cached_texts_embedded": self.cached_texts_embedded,
            "total_cost_usd": self.total_cost_usd,
            "historical_embed_cost_usd": self.historical_embed_cost_usd,
            "total_llm_tokens": self.total_llm_tokens,
            "total_llm_cost_usd": self.total_llm_cost_usd,
            "historical_llm_cost_usd": self.historical_llm_cost_usd,
            "llm_calls": self.llm_calls,
            "llm_cached_pairs": self.llm_cached_pairs,
            "rag_duration_seconds": self.rag_duration_seconds,
            "llm_duration_seconds": self.llm_duration_seconds,
            "llm_model": self.llm_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DuplicateCheckResult":
        """Reconstruct a result previously serialized by to_dict() -- used
        to reload a persisted 'Check Duplicates' run (see trek_cache's
        duplicate_check_runs table) without re-running anything.

        Performance: for large results (9000+ pairs) the original
        ``SimilarityResult(**p)`` approach was slow because Python's
        keyword-argument unpacking + dataclass __init__ for 20 fields
        × 10k pairs adds up. We use object.__new__() + direct __dict__
        assignment to bypass __init__ entirely -- ~3-5x faster."""
        result = cls()
        # Fast-path pair construction: bypass dataclass __init__ overhead
        # by creating instances with __new__ and setting attributes directly.
        # SimilarityResult uses __slots__ (dataclass(slots=True)) so we
        # assign to slots directly via object.__setattr__.
        _SR = SimilarityResult
        _defaults = {
            "syt_text": "", "swt_text": "", "syt_methods": [], "swt_methods": [],
            "score": 0.0, "bm25_score": 0.0, "vector_score": 0.0,
            "sequence_score": 0.0, "has_sequence": False,
            "classification": "distinct", "exact_match": False,
            "llm_verdict": None, "llm_reasoning": "",
            "counterpart_type": "SWT", "syt_tc": {}, "swt_tc": {},
            "llm_cost_usd": 0.0, "checked_at": "",
            "review_status": "", "review_comment": "",
            "reviewer": "", "reviewed_at": "",
        }
        _all_fields = ("syt_id", "swt_id") + tuple(_defaults.keys())
        pairs = []
        for p in data.get("pairs", []):
            obj = _SR.__new__(_SR)
            for f in _all_fields:
                object.__setattr__(obj, f, p.get(f, _defaults.get(f)))
            # Ensure mutable defaults are not shared across instances
            if not p.get("syt_methods"):
                object.__setattr__(obj, "syt_methods", [])
            if not p.get("swt_methods"):
                object.__setattr__(obj, "swt_methods", [])
            if not p.get("syt_tc"):
                object.__setattr__(obj, "syt_tc", {})
            if not p.get("swt_tc"):
                object.__setattr__(obj, "swt_tc", {})
            pairs.append(obj)
        result.pairs = pairs
        result.skipped = data.get("skipped", [])
        result.total_tokens_embedded = data.get("total_tokens_embedded", 0)
        result.total_texts_embedded = data.get("total_texts_embedded", 0)
        result.cached_texts_embedded = data.get("cached_texts_embedded", 0)
        result.total_cost_usd = data.get("total_cost_usd", 0.0)
        result.historical_embed_cost_usd = data.get("historical_embed_cost_usd", 0.0)
        result.total_llm_tokens = data.get("total_llm_tokens", 0)
        result.total_llm_cost_usd = data.get("total_llm_cost_usd", 0.0)
        result.historical_llm_cost_usd = data.get("historical_llm_cost_usd", 0.0)
        result.llm_calls = data.get("llm_calls", 0)
        result.llm_cached_pairs = data.get("llm_cached_pairs", 0)
        result.rag_duration_seconds = data.get("rag_duration_seconds", 0.0)
        result.llm_duration_seconds = data.get("llm_duration_seconds", 0.0)
        result.llm_model = data.get("llm_model", "")
        return result


def merge_duplicate_check_results(old: DuplicateCheckResult, new: DuplicateCheckResult,
                                   run_timestamp: str) -> DuplicateCheckResult:
    """Merge a NEW 'Check Duplicates' run into an OLDER cached result for
    the SAME module+model, so re-running on a smaller/different selection
    of test cases never discards the rest of a larger previously-cached
    list.

    Each pair is keyed by (syt_id, swt_id) -- a NEW pair overwrites the
    OLD pair at that same key (it IS the latest verified result for that
    exact comparison); every OTHER pair already in ``old`` that the new
    run didn't touch is kept as-is. ``run_timestamp`` is stamped onto
    every NEW pair's ``checked_at`` so the GUI can flag which pairs came
    from the latest run within the merged list.

    Aggregate fields (total_cost_usd, total_llm_tokens, llm_calls, ...)
    reflect only THIS run's fresh spend/activity -- they are NOT summed
    across merges, since that would double count if the user runs the
    same subset repeatedly. historical_llm_cost_usd is the one exception:
    it's recomputed as the sum of every pair's OWN llm_cost_usd across the
    WHOLE merged list (old + new), which is exact and never double-counts
    (each pair's cost is tracked once, on that pair, regardless of which
    run computed it) -- unlike embedding cost, which is shared across
    pairs and therefore not tracked per-pair.
    """
    for p in new.pairs:
        p.checked_at = run_timestamp

    by_key: Dict[tuple, SimilarityResult] = {(p.syt_id, p.swt_id): p for p in old.pairs}
    for p in new.pairs:
        by_key[(p.syt_id, p.swt_id)] = p

    merged = DuplicateCheckResult(pairs=list(by_key.values()))
    merged.skipped = new.skipped
    merged.total_tokens_embedded = new.total_tokens_embedded
    merged.total_texts_embedded = new.total_texts_embedded
    merged.cached_texts_embedded = new.cached_texts_embedded
    merged.total_cost_usd = new.total_cost_usd
    merged.total_llm_tokens = new.total_llm_tokens
    merged.total_llm_cost_usd = new.total_llm_cost_usd
    merged.llm_calls = new.llm_calls
    merged.llm_cached_pairs = new.llm_cached_pairs
    merged.rag_duration_seconds = new.rag_duration_seconds
    merged.llm_duration_seconds = new.llm_duration_seconds
    merged.llm_model = new.llm_model
    merged.historical_embed_cost_usd = new.historical_embed_cost_usd
    merged.historical_llm_cost_usd = sum(p.llm_cost_usd for p in merged.pairs)
    return merged


@dataclass
class EmbeddingUsage:
    """Accumulates EXACT token/cost usage across one or more embed calls,
    read from the gateway's own response (usage.total_tokens and the
    x-litellm-response-cost header) rather than estimated locally -- see
    _embed_batch_with_retry(). cost stays 0.0 if the gateway doesn't
    report the cost header for embedding requests."""
    tokens: int = 0
    cost: float = 0.0


def get_embeddings_client(jwt_token: str):
    """Construct a raw OpenAI SDK client pointed at the internal LLM
    gateway, using the exact same construction pattern as
    base_organization_retriever.BaseOrganizationRetriever.get_embeddings()
    (same headers) -- WITHOUT that method's blocking check_instance()
    polling loop, since a GUI action should fail fast with a clear error
    rather than silently hang for up to 4 minutes per retry if the
    gateway is unreachable.

    Deliberately NOT langchain_openai.OpenAIEmbeddings: that wrapper
    hides the raw HTTP response, so callers can't read the gateway's
    reported token usage / cost. Using the raw SDK client's
    ``embeddings.with_raw_response.create(...)`` (see
    _embed_batch_with_retry()) gives EXACT token counts and cost instead
    of the char-based estimate this module previously had to rely on --
    mirrors the same with_raw_response pattern already used for chat
    completions in openai_api.request_OpenAI[_sync]().

    The gateway URL is hardcoded (LLM_GATEWAY_URL, above) -- discovering
    duplicate SYT/SWT test cases is this application's core purpose, not
    an optional add-on, so there is exactly one gateway to talk to and no
    reason to make it user-configurable. The JWT token, however, is a
    per-user credential and MUST be supplied explicitly by the caller
    (see trek_projects.TrekProjectStore -- every project requires one).

    Args:
        jwt_token: the caller's JWT token for the gateway. Required --
                   this function does not fall back to an environment
                   variable or any other implicit source.

    Raises:
        RuntimeError: if jwt_token is empty/None.
        ImportError: if the `openai` package is not installed in this environment.
    """
    if not jwt_token:
        raise RuntimeError(
            "No JWT Token configured for this project. Set it via "
            "'Edit Project' (header ✎ button) -- a JWT Token is required "
            "for every project since duplicate detection is this "
            "application's core purpose."
        )

    from openai import OpenAI

    return OpenAI(
        base_url=f"{LLM_GATEWAY_URL}/llm",
        api_key=jwt_token,
        default_headers={
            "X-Application-Name": "MAIA",
            "X-Application-Cusco-Id": "1000192",
            "X-Application-Token": "gAAAAABqeXFaYRNVV6Ozi0sAptlR7t1hKU7RfBGRLl18NcyDoYVoPezy1jw6qPqUo0ovh0rGvGx1CBQwFEv3tvAOjiUTDkFSVP7Q6ekHQ_p23Yce1OdZTYw=",
        },
    )


MAX_TOKENS_PER_EMBED_REQUEST = 20000

# Conservative token-estimation heuristic: assume 1 token ~= 3 characters
# (English technical/procedural text tends to run closer to 4 chars/token,
# but test case text is full of short identifiers, punctuation, and
# numbers which tokenize less efficiently -- erring conservative means we
# chunk a bit more aggressively than strictly necessary rather than risk
# still exceeding the real limit).
_CHARS_PER_TOKEN_ESTIMATE = 3

# Real-world failure this margin fixes: a batch whose char/3 estimate
# came out at <=20000 (MAX_TOKENS_PER_EMBED_REQUEST) was still rejected by
# Vertex AI with "the input token count is 21223 but the model supports
# up to 20000" -- a ~6% underestimate, because char/3 is only a cheap
# heuristic (no real tokenizer is available here) and test-case text's
# mix of identifiers/punctuation/numbers doesn't tokenize uniformly.
# Splitting batches at a threshold well BELOW the real hard limit (rather
# than splitting right up against it) absorbs estimation error of this
# kind before it ever reaches the API. Combined with the adaptive
# retry-with-halving in _embed_batch_with_retry() below (which reacts to
# the API's own real token count if our estimate is ever STILL wrong),
# embed_texts() is now robust to estimation error of any magnitude, not
# just the specific ~6% observed in this one incident.
_BATCH_SAFETY_MARGIN_RATIO = 0.75
BATCH_TOKEN_BUDGET = int(MAX_TOKENS_PER_EMBED_REQUEST * _BATCH_SAFETY_MARGIN_RATIO)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN_ESTIMATE
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _is_token_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a token-limit-exceeded error from the
    embedding gateway, e.g. Vertex AI's 400 INVALID_ARGUMENT:
    "Unable to submit request because the input token count is 21223 but
    the model supports up to 20000." Matched loosely on keywords rather
    than a specific exception type, since the underlying error can arrive
    wrapped in different exception classes depending on the HTTP client /
    langchain version in use.
    """
    msg = str(exc).lower()
    return "token" in msg and ("count" in msg or "limit" in msg or "supports up to" in msg)


def _embed_batch_with_retry(client, batch: List[str], depth: int = 0,
                             usage: Optional[EmbeddingUsage] = None,
                             per_text_cost: Optional[Dict[str, float]] = None) -> List[List[float]]:
    """Embed one batch, automatically halving-and-retrying if the API
    itself rejects the batch as exceeding the real token limit.

    Uses ``embeddings.with_raw_response.create(...)`` (rather than a
    langchain wrapper) so the gateway's OWN reported token usage
    (``response.usage.total_tokens``) and cost (``x-litellm-response-cost``
    header) can be read directly and accumulated into ``usage`` --
    exact figures, not the char-based estimate used elsewhere in this
    module for pre-flight batch sizing.

    This is the safety net of last resort: BATCH_TOKEN_BUDGET's margin
    (see above) is meant to prevent this from ever triggering in
    practice, but if the character-based token estimate is ever wrong by
    MORE than that margin -- for any reason, including content mixes we
    haven't seen yet -- this reacts to the API's own authoritative token
    count instead of trusting our estimate, splitting the offending batch
    in half and retrying each half independently (recursively, if
    needed) rather than failing the entire "Check Duplicates" run over
    one oversized batch.

    If a batch has already been reduced to a single text and the API
    STILL rejects it, that single text is truncated further (halved) and
    retried, since it can't be split into smaller batches anymore.
    ``depth`` bounds the recursion (a single text can only be halved so
    many times before becoming empty) to guarantee this always
    terminates rather than looping forever against a persistently
    failing gateway.

    Args:
        per_text_cost: optional dict populated with {text: cost_usd},
                       splitting each successful call's real reported
                       cost evenly across the texts in that specific
                       batch -- lets callers persist a REAL per-text
                       cost (e.g. into trek_cache's embeddings table) for
                       later historical-cost reporting, not an estimate.
    """
    try:
        raw = client.embeddings.with_raw_response.create(input=batch, model=EMBEDDING_MODEL)
        cost = float(raw.headers.get("x-litellm-response-cost", 0) or 0)
        response = raw.parse()
        if not cost:
            # Gateway didn't report a cost for this model -- fall back to
            # the known per-token price (see EMBEDDING_MODEL_PRICING).
            cost = response.usage.total_tokens * EMBEDDING_MODEL_PRICING.get(EMBEDDING_MODEL, 0.0)
        if usage is not None:
            usage.tokens += response.usage.total_tokens
            usage.cost += cost
        if per_text_cost is not None and batch:
            cost_per_item = cost / len(batch)
            for text in batch:
                per_text_cost[text] = cost_per_item
        return [item.embedding for item in response.data]
    except Exception as exc:
        if not _is_token_limit_error(exc) or depth >= 8:
            raise
        if len(batch) == 1:
            text = batch[0]
            truncated = text[: max(1, len(text) // 2)]
            result = _embed_batch_with_retry(client, [truncated], depth + 1, usage, per_text_cost)
            if per_text_cost is not None and truncated in per_text_cost:
                per_text_cost[text] = per_text_cost[truncated]
            return result
        mid = len(batch) // 2
        left = _embed_batch_with_retry(client, batch[:mid], depth + 1, usage, per_text_cost)
        right = _embed_batch_with_retry(client, batch[mid:], depth + 1, usage, per_text_cost)
        return left + right


def embed_texts(client, texts: List[str],
                 max_tokens_per_request: int = BATCH_TOKEN_BUDGET,
                 usage: Optional[EmbeddingUsage] = None,
                 per_text_cost: Optional[Dict[str, float]] = None) -> List[List[float]]:
    """Batch-embed a list of texts, automatically splitting into multiple
    API calls so no single request exceeds the embedding model's token
    limit. When ``usage`` is provided, it accumulates the EXACT token
    count and cost reported by the gateway for every sub-batch actually
    sent (see _embed_batch_with_retry()). When ``per_text_cost`` is
    provided, it is populated with a REAL per-text cost estimate (each
    batch's real cost split evenly across its texts) for historical-cost
    reporting later.

    Real-world failure this fixes: embedding all of a large "Check
    Duplicates" run's texts in one client.embed_documents(texts) call can
    trivially exceed the model's per-request token budget (e.g. 20,000)
    once dozens of full test-case texts are concatenated into one request
    -- Vertex AI then rejects the ENTIRE batch with a 400
    INVALID_ARGUMENT error, silently failing every pair in the run rather
    than just the oversized one.

    Three safeguards, from cheapest/most-common to last-resort:
      1. Any SINGLE text that individually exceeds the budget is
         truncated to fit (better to compare a truncated-but-present text
         than to abort the whole run over one outlier).
      2. Texts are grouped into sub-batches that each stay under
         BATCH_TOKEN_BUDGET -- a threshold deliberately set BELOW the
         real model limit (MAX_TOKENS_PER_EMBED_REQUEST) to absorb error
         in the character-based token estimate (no real tokenizer is
         available here) -- and each sub-batch is sent as its own
         embed_documents() call.
      3. If the API itself STILL rejects a sub-batch as too large despite
         the margin above (i.e. our estimate was wrong by more than the
         margin), _embed_batch_with_retry() automatically halves that
         specific batch and retries, recursively, rather than failing the
         entire run.
    """
    if not texts:
        return []

    safe_texts = [
        _truncate_to_token_budget(t, max_tokens_per_request) if _estimate_tokens(t) > max_tokens_per_request
        else t
        for t in texts
    ]

    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0
    for text in safe_texts:
        text_tokens = _estimate_tokens(text)
        if current_batch and current_tokens + text_tokens > max_tokens_per_request:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += text_tokens
    if current_batch:
        batches.append(current_batch)

    results: List[List[float]] = []
    for batch in batches:
        results.extend(_embed_batch_with_retry(client, batch, usage=usage, per_text_cost=per_text_cost))
    return results


def cosine_similarity_matrix(vectors_a, vectors_b):
    """Compute the full pairwise cosine similarity matrix between two sets
    of vectors, using the same L2-normalize-then-dot-product pattern as
    ClusterRAGOrchestrator._select_representatives() (`sim = mat @ mat.T`
    there is the same operation applied to a single set against itself).

    Args:
        vectors_a: list of embedding vectors, shape (n, d)
        vectors_b: list of embedding vectors, shape (m, d)

    Returns:
        numpy.ndarray of shape (n, m) with cosine similarities in [-1, 1]
        (in practice [0, 1] for text embeddings of this kind).
    """
    import numpy as np

    a = np.array(vectors_a, dtype=float)
    b = np.array(vectors_b, dtype=float)

    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    # Avoid division by zero for any degenerate all-zero embedding.
    a_norms[a_norms == 0] = 1.0
    b_norms[b_norms == 0] = 1.0

    a_normalized = a / a_norms
    b_normalized = b / b_norms

    return a_normalized @ b_normalized.T


class BM25CorpusScorer:
    """Wraps rank_bm25.BM25Okapi over a fixed corpus of texts, providing a
    normalized-to-[0,1] pairwise lexical similarity score between any two
    texts already in the corpus.

    Unlike the RAG codebase's usage (one query against a catalog of
    candidate documents), duplicate detection compares two SPECIFIC known
    texts. To make BM25's IDF weighting meaningful (rare/distinctive terms
    matter more than common ones), the corpus is built from ALL unique
    texts in the current comparison batch -- e.g. every SYT and SWT test
    case text being checked in this run -- so common boilerplate phrases
    (present in most test cases) get naturally down-weighted relative to
    distinctive content.

    BM25 is asymmetric (query vs. document); since duplicate detection has
    no natural "query" side, `pair_score()` computes both directions
    (text_a-as-query-against-text_b, and vice versa) and returns their
    average for a direction-independent similarity signal.
    """

    def __init__(self, texts: List[str]):
        from rank_bm25 import BM25Okapi

        self.texts = list(dict.fromkeys(texts))   # de-duplicate, preserve order
        self._index_by_text = {t: i for i, t in enumerate(self.texts)}
        self._tokenized_corpus = [_tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

        if self._bm25 is not None:
            # rank_bm25's classic IDF formula, log((N - n + 0.5) / (n + 0.5)),
            # goes NEGATIVE for any term appearing in more than half the
            # corpus -- entirely plausible here since the corpus is small
            # (built from just the current comparison batch) and duplicate
            # SYT/SWT pairs by definition share lots of common words. A
            # negative IDF means "this term appearing in both documents
            # actively REDUCES their similarity score", which is backwards
            # for duplicate detection (shared common words should still
            # count as some evidence of similarity, just weighted less than
            # rare/distinctive words). Clamping negative IDF to a small
            # positive epsilon is the standard fix (same one Lucene/Elastic-
            # search apply) and keeps every raw score non-negative so the
            # max-score normalization below never degenerates to all-zeros.
            self._bm25.idf = {
                term: (value if value > 0 else 1e-4)
                for term, value in self._bm25.idf.items()
            }

    def _normalized_scores_for_query(self, query_text: str):
        """Return BM25 scores for `query_text` against every corpus
        document, normalized to [0,1] by dividing by the max raw score
        (mirrors ANSRetriever._get_bm25_internals()'s normalization)."""
        import numpy as np

        if self._bm25 is None:
            return np.array([])
        query_tokens = _tokenize(query_text)
        raw_scores = self._bm25.get_scores(query_tokens)
        max_score = raw_scores.max() if len(raw_scores) else 0.0
        if max_score <= 0:
            return np.zeros_like(raw_scores)
        return raw_scores / max_score

    def pair_score(self, text_a: str, text_b: str) -> float:
        """Symmetric normalized BM25 similarity between two texts already
        present in this scorer's corpus. Returns 0.0 if either text isn't
        in the corpus (shouldn't happen if the corpus was built correctly
        by the caller) or the corpus is empty."""
        idx_a = self._index_by_text.get(text_a)
        idx_b = self._index_by_text.get(text_b)
        if idx_a is None or idx_b is None or self._bm25 is None:
            return 0.0

        scores_a_as_query = self._normalized_scores_for_query(text_a)
        scores_b_as_query = self._normalized_scores_for_query(text_b)
        score_a_to_b = float(scores_a_as_query[idx_b]) if len(scores_a_as_query) else 0.0
        score_b_to_a = float(scores_b_as_query[idx_a]) if len(scores_b_as_query) else 0.0
        return (score_a_to_b + score_b_to_a) / 2.0


def ensemble_score(bm25_score: float, vector_score: float,
                    bm25_weight: float = DEFAULT_BM25_WEIGHT,
                    vec_weight: float = DEFAULT_VEC_WEIGHT,
                    sequence_score: Optional[float] = None,
                    seq_weight: float = DEFAULT_SEQ_WEIGHT) -> float:
    """Combine BM25, vector, and (optionally) method-call sequence scores
    into a single ensemble score via a straight weighted average.

    When `sequence_score` is None (at least one side of the pair had no
    scripted Component.method(...) procedure to compare -- e.g. a pure
    prose/manual test case), the sequence dimension is EXCLUDED from the
    ensemble entirely and the remaining bm25_weight/vec_weight are
    renormalized to sum to 1.0, rather than treating a missing signal as
    0.0 (which would unfairly penalize non-scripted test cases just for
    lacking a procedure to compare).

    NOTE: earlier versions of this function copied
    ANSRetriever.retrieve_with_scores()'s fallback rule ("if BM25 < 0.3,
    ignore it and use the vector score alone"). That rule makes sense in
    that codebase's RETRIEVAL context -- ranking many candidate documents
    against a query, where a document with zero lexical overlap might
    still be the best available match. It is WRONG for duplicate
    DETECTION between two specific known texts: it silently discarded
    whatever weight the user configured every time BM25 was weak, which
    for full test-case text (not short queries) is nearly always the case
    -- case-specific details (values, variable names, method arguments)
    naturally dilute lexical overlap even between genuinely duplicate
    test cases. The user's configured weights are always honored here
    (aside from the documented sequence-availability renormalization
    above), with no silent per-score override.
    """
    if sequence_score is None:
        total = bm25_weight + vec_weight
        if total <= 0:
            return vector_score
        return (bm25_weight / total) * bm25_score + (vec_weight / total) * vector_score
    return bm25_weight * bm25_score + vec_weight * vector_score + seq_weight * sequence_score


def build_bare_pairs(pairs: List[tuple]) -> "DuplicateCheckResult":
    """Build a DuplicateCheckResult WITHOUT running the hybrid BM25/vector/
    sequence scoring pipeline (see compare_syt_swt_pairs()) -- used when the
    user unchecks 'Run RAG scoring' in the settings dialog to skip embedding
    calls entirely and go straight to the LLM-judge stage.

    Each pair gets classification="not_scored" (score=0.0, not a real
    ensemble result) so the GUI clearly shows this dimension wasn't
    evaluated, rather than defaulting to a misleading "distinct".
    """
    result = DuplicateCheckResult()
    for syt_id, syt_text, swt_id, swt_text, syt_methods, swt_methods, counterpart_type, *tc_dicts in pairs:
        syt_tc, swt_tc = (tc_dicts[0], tc_dicts[1]) if len(tc_dicts) == 2 else ({}, {})
        if not syt_text or not swt_text:
            if not syt_text:
                result.skipped.append(syt_id)
            if not swt_text:
                result.skipped.append(swt_id)
            continue
        result.pairs.append(SimilarityResult(
            syt_id=syt_id, swt_id=swt_id, syt_text=syt_text, swt_text=swt_text,
            syt_methods=syt_methods, swt_methods=swt_methods,
            classification="not_scored", counterpart_type=counterpart_type,
            syt_tc=syt_tc, swt_tc=swt_tc,
        ))
    return result


# Per-token pricing for the LLM-judge stage's cost ESTIMATE (shown before a
# run, in $), kept in sync with vehicleConvert's src/utils/token_stats.py
# MODEL_PRICING -- that's the authoritative pricing table this whole
# LLM Incubator gateway family already uses for cost accounting elsewhere.
# NOTE: this is an ESTIMATE based on average text length, not the exact
# post-run cost (see DuplicateCheckResult.total_llm_cost_usd for that,
# read directly from the gateway's own response).
MODEL_PRICING = {
    # ── GPT ────────────────────────────────────────────────────
    "gpt-5-mini":       {"input_cost": 0.25 / 1_000_000, "output_cost":  2.00 / 1_000_000},
    "gpt-5-nano":       {"input_cost": 0.10 / 1_000_000, "output_cost":  0.40 / 1_000_000},
    "gpt-5":            {"input_cost": 1.25 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5-chat":       {"input_cost": 1.25 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5.1":          {"input_cost": 1.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5.1-chat":     {"input_cost": 1.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5.2":          {"input_cost": 1.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5.2-chat":     {"input_cost": 1.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-5.3-codex":    {"input_cost": 2.00 / 1_000_000, "output_cost": 12.00 / 1_000_000},
    "gpt-5.4":          {"input_cost": 2.50 / 1_000_000, "output_cost": 15.00 / 1_000_000},
    "gpt-5.4-mini":     {"input_cost": 0.30 / 1_000_000, "output_cost":  2.50 / 1_000_000},
    "gpt-5.4-nano":     {"input_cost": 0.10 / 1_000_000, "output_cost":  0.40 / 1_000_000},
    "gpt-5.5":          {"input_cost": 3.00 / 1_000_000, "output_cost": 15.00 / 1_000_000},
    "gpt-5.6-luna":     {"input_cost": 3.00 / 1_000_000, "output_cost": 15.00 / 1_000_000},
    "gpt-5.6-sol":      {"input_cost": 3.00 / 1_000_000, "output_cost": 15.00 / 1_000_000},
    "gpt-5.6-terra":    {"input_cost": 3.00 / 1_000_000, "output_cost": 15.00 / 1_000_000},
    "gpt-4o":           {"input_cost": 2.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gpt-4o-mini":      {"input_cost": 0.15 / 1_000_000, "output_cost":  0.60 / 1_000_000},
    "gpt-35-turbo":     {"input_cost": 0.50 / 1_000_000, "output_cost":  1.50 / 1_000_000},
    "gpt-o3-mini":      {"input_cost": 1.10 / 1_000_000, "output_cost":  4.40 / 1_000_000},
    # ── Claude ─────────────────────────────────────────────────
    "vertex_ai/claude-opus-5":   {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-opus-4-6":           {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-opus-4-5":           {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-opus-5:reasoning":   {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-4-6-opus-v1:0":      {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-4-7-opus-v1:0":      {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    "claude-4-8-opus-v1:0":      {"input_cost": 15.00 / 1_000_000, "output_cost": 75.00 / 1_000_000},
    # ── Gemini ─────────────────────────────────────────────────
    "gemini-2.5-pro":            {"input_cost": 1.25 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gemini-2.5-flash":          {"input_cost": 0.15 / 1_000_000, "output_cost":  0.60 / 1_000_000},
    "gemini-2.5-flash-lite":     {"input_cost": 0.08 / 1_000_000, "output_cost":  0.30 / 1_000_000},
    "failover/gemini-2.5-pro":   {"input_cost": 1.25 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gemini-3.1-pro-preview":    {"input_cost": 1.50 / 1_000_000, "output_cost": 10.00 / 1_000_000},
    "gemini-3.1-flash-lite":     {"input_cost": 0.08 / 1_000_000, "output_cost":  0.30 / 1_000_000},
    "gemini-3.1-flash-image-preview": {"input_cost": 0.15 / 1_000_000, "output_cost": 0.60 / 1_000_000},
    "gemini-3.5-flash":          {"input_cost": 0.15 / 1_000_000, "output_cost":  0.60 / 1_000_000},
    "gemini-3.5-flash-lite":     {"input_cost": 0.08 / 1_000_000, "output_cost":  0.30 / 1_000_000},
    "gemini-3.6-flash":          {"input_cost": 0.15 / 1_000_000, "output_cost":  0.60 / 1_000_000},
    "gemini-3.7-flash":          {"input_cost": 0.15 / 1_000_000, "output_cost":  0.60 / 1_000_000},
    # ── Llama ──────────────────────────────────────────────────
    "meta-llama/Llama-3.3-70B-Instruct": {"input_cost": 0.72 / 1_000_000, "output_cost": 0.72 / 1_000_000},
}

_ESTIMATE_CHARS_PER_TOKEN = 4        # rough English-prose ratio for a pre-run estimate
_ESTIMATE_OUTPUT_TOKENS_PER_PAIR = 40  # a verdict + one-sentence reasoning is short


def estimate_llm_judge_cost(num_pairs: int, model: str, batch_size: int,
                             instructions: str = "", avg_pair_chars: int = 1200,
                             num_groups: Optional[int] = None) -> Optional[float]:
    """Rough PRE-RUN cost estimate (in USD) for running the LLM-judge stage
    over ``num_pairs`` pairs, given the batching parameters that will
    actually be used. Returns None if ``model`` isn't in MODEL_PRICING (an
    unpriced/unknown model -- can't estimate).

    This is intentionally approximate (char/4 token heuristic, no real
    tokenizer) -- it exists so the user can see a ballpark BEFORE spending
    real money, not to replace the exact post-run cost in
    DuplicateCheckResult.total_llm_cost_usd (read directly from the
    gateway's response). Larger batch_size lowers the estimate because the
    system-prompt/instructions overhead is only sent once per BATCH, not
    once per pair -- reflecting the real reason batching is cheaper.

    Args:
        num_pairs: total pairs that will be judged.
        model: chat model name (looked up in MODEL_PRICING).
        batch_size: SYT groups (see run_llm_judge_stage()) judged per
                    single LLM request.
        instructions: the actual judging-instructions text that will be
                      sent (defaults to LLM_JUDGE_DEFAULT_INSTRUCTIONS's
                      length if blank), so the estimate reflects custom
                      instructions the user has typed.
        avg_pair_chars: average combined SYT+SWT text length per pair --
                        pass the real average from the actual traceability
                        rows when available for a much better estimate.
        num_groups: distinct SYT groups the pairs belong to -- batching is
                    now SYT-group-based, not raw-pair-based (see
                    run_llm_judge_stage()), so this determines the actual
                    number of requests. Falls back to num_pairs (one group
                    per pair) if not provided.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None or num_pairs <= 0:
        return None

    criteria = instructions.strip() if instructions and instructions.strip() else LLM_JUDGE_DEFAULT_INSTRUCTIONS
    system_prompt_tokens = (len(criteria) + len(_LLM_JUDGE_OUTPUT_FORMAT)) // _ESTIMATE_CHARS_PER_TOKEN
    groups = num_groups if num_groups is not None else num_pairs
    num_batches = -(-groups // max(1, batch_size))  # ceil division

    input_tokens = (
        num_batches * system_prompt_tokens
        + num_pairs * (avg_pair_chars // _ESTIMATE_CHARS_PER_TOKEN)
    )
    output_tokens = num_pairs * _ESTIMATE_OUTPUT_TOKENS_PER_PAIR

    return input_tokens * pricing["input_cost"] + output_tokens * pricing["output_cost"]


def compare_syt_swt_pairs(
    client,
    pairs: List[tuple],
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    duplicate_threshold: float = SIMILARITY_DUPLICATE,
    near_duplicate_threshold: float = SIMILARITY_NEAR_DUPLICATE,
    similar_threshold: float = SIMILARITY_SIMILAR,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vec_weight: float = DEFAULT_VEC_WEIGHT,
    seq_weight: float = DEFAULT_SEQ_WEIGHT,
    per_text_cost: Optional[Dict[str, float]] = None,
) -> DuplicateCheckResult:
    """Compare a list of (syt_id, syt_text, swt_id, swt_text, syt_methods,
    swt_methods) tuples and return a DuplicateCheckResult with one
    SimilarityResult per pair, scored via the hybrid BM25 + vector +
    method-call-sequence ensemble formula. syt_methods/swt_methods are the
    ordered method-call lists from extract_method_call_sequence() -- pass
    empty lists for test cases with no scripted procedure (the sequence
    dimension is then excluded from that pair's ensemble automatically,
    see ensemble_score()).

    Pipeline:
      1. Exact-match pre-pass on normalized text -- free, no embedding or
         BM25 call needed for pairs that are literal copies of each other.
      2. Build a single BM25Okapi corpus from every unique remaining text
         in the batch (see BM25CorpusScorer), so lexical scoring has
         meaningful IDF weighting.
      3. Embed all texts NOT already resolved by stage 1 (deduplicating
         identical strings across the whole batch so each unique text is
         only embedded once, and reusing `embedding_cache` for texts seen
         in a previous run) and compute cosine similarity per pair.
      4. Compute order-sensitive method-call sequence similarity per pair
         (only when both sides have a non-empty sequence).
      5. Combine BM25 + vector + sequence scores into a final ensemble
         score per pair via `ensemble_score()`, then classify against the
         given thresholds.

    Args:
        client: an embeddings client from get_embeddings_client() (or a
                test double exposing the same embed_documents() method).
        pairs: list of (syt_id, syt_text, swt_id, swt_text, syt_methods, swt_methods, counterpart_type).
        embedding_cache: optional dict of normalized_text -> embedding
                         vector, reused/updated in place so callers can
                         persist it (e.g. to trek_cache.sqlite3) across runs.
        duplicate_threshold / near_duplicate_threshold / similar_threshold:
                         classification band cutoffs on the ENSEMBLE score.
        bm25_weight / vec_weight / seq_weight: hybrid fusion weights
                         (should sum to 1.0, but not enforced here --
                         validated in the GUI dialog).
        per_text_cost: optional dict populated with {text: cost_usd} for
                       every NEWLY embedded text this call (cache hits
                       excluded) -- lets callers persist a REAL per-text
                       cost for historical-cost reporting later (see
                       trek_cache.TrekCache.set_embeddings()).
    """
    embedding_cache = embedding_cache if embedding_cache is not None else {}
    result = DuplicateCheckResult()

    # Stage 1: exact-match pre-pass; collect remaining texts to embed/score.
    seen_norm_texts = set(embedding_cache.keys())
    texts_to_embed: List[str] = []
    # (syt_id, syt_text, swt_id, swt_text, norm_syt, norm_swt, syt_methods, swt_methods, counterpart_type, syt_tc, swt_tc)
    pending: List[tuple] = []

    for syt_id, syt_text, swt_id, swt_text, syt_methods, swt_methods, counterpart_type, *tc_dicts in pairs:
        syt_tc, swt_tc = (tc_dicts[0], tc_dicts[1]) if len(tc_dicts) == 2 else ({}, {})
        if not syt_text or not swt_text:
            if not syt_text:
                result.skipped.append(syt_id)
            if not swt_text:
                result.skipped.append(swt_id)
            continue

        norm_syt = _normalize_for_exact_match(syt_text)
        norm_swt = _normalize_for_exact_match(swt_text)

        if norm_syt == norm_swt:
            result.pairs.append(SimilarityResult(
                syt_id=syt_id, swt_id=swt_id, syt_text=syt_text, swt_text=swt_text,
                syt_methods=syt_methods, swt_methods=swt_methods,
                score=1.0, bm25_score=1.0, vector_score=1.0, sequence_score=1.0,
                has_sequence=bool(syt_methods and swt_methods),
                classification="duplicate", exact_match=True, counterpart_type=counterpart_type,
                syt_tc=syt_tc, swt_tc=swt_tc,
            ))
            continue

        pending.append((syt_id, syt_text, swt_id, swt_text, norm_syt, norm_swt, syt_methods, swt_methods, counterpart_type, syt_tc, swt_tc))
        for norm_text, raw_text in ((norm_syt, syt_text), (norm_swt, swt_text)):
            if norm_text not in seen_norm_texts:
                texts_to_embed.append(raw_text)
                seen_norm_texts.add(norm_text)

    if not pending:
        return result

    # Stage 2: build the BM25 corpus from every unique pending text.
    all_pending_texts: List[str] = []
    for _syt_id, syt_text, _swt_id, swt_text, _ns, _nw, _sm, _wm, _ct, _stc, _wtc in pending:
        all_pending_texts.append(syt_text)
        all_pending_texts.append(swt_text)
    bm25_scorer = BM25CorpusScorer(all_pending_texts)

    # Stage 3: embed whatever wasn't already cached.
    if texts_to_embed:
        usage = EmbeddingUsage()
        new_vectors = embed_texts(client, texts_to_embed, usage=usage, per_text_cost=per_text_cost)
        result.total_texts_embedded = len(texts_to_embed)
        result.total_tokens_embedded = usage.tokens
        result.total_cost_usd = usage.cost
        for text, vec in zip(texts_to_embed, new_vectors):
            embedding_cache[_normalize_for_exact_match(text)] = vec

    # Stage 4/5: compute sequence similarity + score every pending pair.
    for syt_id, syt_text, swt_id, swt_text, norm_syt, norm_swt, syt_methods, swt_methods, counterpart_type, syt_tc, swt_tc in pending:
        vec_syt = embedding_cache.get(norm_syt)
        vec_swt = embedding_cache.get(norm_swt)
        if vec_syt is None or vec_swt is None:
            result.skipped.append(syt_id if vec_syt is None else swt_id)
            continue

        vector_score = float(cosine_similarity_matrix([vec_syt], [vec_swt])[0][0])
        bm25_score = bm25_scorer.pair_score(syt_text, swt_text)

        sequence_score = (
            method_sequence_similarity(syt_methods, swt_methods)
            if syt_methods and swt_methods else None
        )

        final_score = ensemble_score(
            bm25_score, vector_score, bm25_weight, vec_weight, sequence_score, seq_weight
        )
        classification = classify_similarity(
            final_score, duplicate_threshold, near_duplicate_threshold, similar_threshold
        )
        result.pairs.append(SimilarityResult(
            syt_id=syt_id, swt_id=swt_id, syt_text=syt_text, swt_text=swt_text,
            syt_methods=syt_methods, swt_methods=swt_methods,
            score=final_score, bm25_score=bm25_score, vector_score=vector_score,
            sequence_score=sequence_score if sequence_score is not None else 0.0,
            has_sequence=sequence_score is not None,
            classification=classification, exact_match=False, counterpart_type=counterpart_type,
            syt_tc=syt_tc, swt_tc=swt_tc,
        ))

    return result


# ---------------------------------------------------------------------------
# Optional LLM-judge stage -- reads both test cases' actual text/logic
# ---------------------------------------------------------------------------

# Default judging criteria, shown (and editable) in the GUI's 'Check
# Duplicates' settings dialog -- unlike the old private/hidden prompt,
# this is a public constant so users can see exactly what the LLM is
# asked to do by default, and override it entirely per-run if they want
# different criteria (e.g. "ignore ECU/variant naming differences").
LLM_JUDGE_DEFAULT_INSTRUCTIONS = (
    "You are a senior test engineer reviewing automotive requirements test "
    "cases for redundant coverage. You will be given a SYSTEM-level test "
    "case (SYT) together with ONE of two kinds of counterpart: either a "
    "SOFTWARE-level test case (SWT) that refines it, OR another SYSTEM-"
    "level test case (a \"related SYT\") that shares the same system "
    "requirement (SYR) -- the pair type is labeled in each pair below.\n\n"
    "IGNORE surface-level differences that don't change what is actually "
    "being verified: vocabulary, formatting, step ordering that doesn't "
    "affect the outcome, variable/signal naming conventions, and DOORS/"
    "TREK scripting boilerplate (setup calls, breakpoints, internal signal "
    "names). Judge on FUNCTIONAL intent only -- what TRIGGER/PRECONDITION "
    "is applied, and what EXPECTED RESULT/BEHAVIOR is checked.\n\n"
    "Use these definitions:\n"
    "- same_scenario: the counterpart verifies the IDENTICAL trigger and "
    "expected result as the SYT, at essentially the same level of detail "
    "-- for an SWT pair this means the software test adds no real "
    "verification value beyond the system test; for a related-SYT pair "
    "this means the two system tests duplicate each other rather than "
    "cover distinct aspects of the shared requirement. Reversing step "
    "order or renaming identical values (e.g. \"Code4 then Code5\" vs "
    "\"Code5 then Code4\") is still same_scenario.\n"
    "- partial_overlap: the counterpart shares the same general trigger "
    "or requirement area but meaningfully differs in scope, depth, or the "
    "specific values/conditions verified -- e.g. it checks an ADDITIONAL "
    "consequence beyond the SYT, verifies the same behavior through a "
    "different MECHANISM (e.g. an internal software signal vs. an "
    "external observable output), or only covers PART of what the SYT "
    "covers.\n"
    "- different_scenario: the counterpart exercises a genuinely "
    "different trigger, fault type, boundary value, mode, or requirement "
    "aspect -- even if the overall test STRUCTURE looks similar. "
    "Different fault types (e.g. short-to-ground vs. short-to-battery), "
    "different boundary/limit values, or different preconditions/power "
    "modes normally count as different_scenario, since they exercise a "
    "distinct code path or requirement branch, not just a cosmetic "
    "variation of the same one.\n\n"
    "This is safety-critical automotive test coverage: when genuinely "
    "unsure between same_scenario and partial_overlap, prefer "
    "partial_overlap -- flagging a pair for human review is cheap, but "
    "wrongly marking distinct coverage as fully redundant risks a real "
    "test gap if that coverage is ever removed. Ground every verdict in a "
    "SPECIFIC difference or similarity you can point to in the text, not "
    "a general impression."
)

# Output-format contract -- always appended after the (possibly user-edited)
# instructions above, never itself editable, so JSON parsing never breaks
# regardless of what the user changes. Requests judgment on a NUMBERED
# BATCH of pairs at once (rather than one pair per request) so a full
# duplicate-check run needs far fewer LLM calls -- e.g. 277 pairs at the
# default batch size of 15 is ~19 requests instead of 277.
_LLM_JUDGE_OUTPUT_FORMAT = (
    "\n\nYou will be given multiple numbered SYT/SWT pairs. Judge EACH pair "
    "independently using the criteria above.\n\n"
    "Respond with ONLY a JSON object, no other text, of this exact form:\n"
    '{"results": [{"pair": <pair number>, "verdict": "same_scenario" | '
    '"partial_overlap" | "different_scenario", "reasoning": "one sentence"}, '
    '...]}\n'
    "Include EXACTLY one entry per pair number given, in any order."
)


def _build_llm_judge_messages(pairs: List["tuple[str, str, str]"], instructions: str = "") -> List[dict]:
    """Build the judge prompt for a BATCH of (syt_text, counterpart_text,
    counterpart_type) pairs, numbered 1..N in the user message, from the
    (possibly user-edited) judging instructions plus the fixed batch
    output-format contract. Falls back to LLM_JUDGE_DEFAULT_INSTRUCTIONS
    when ``instructions`` is blank.

    counterpart_type is "SWT" (software test case) or "related_syt" (a
    DIFFERENT system test case that shares a SYR with this one) -- labeled
    accordingly so the model understands what it's actually comparing.
    """
    criteria = instructions.strip() if instructions and instructions.strip() else LLM_JUDGE_DEFAULT_INSTRUCTIONS
    system_prompt = criteria + _LLM_JUDGE_OUTPUT_FORMAT
    pairs_text = "\n\n".join(
        f"Pair {i}:\nSYT (system test case):\n{syt_text}\n\n"
        f"{'SWT (software test case)' if counterpart_type != 'related_syt' else 'Related SYT (another system test case)'}:\n{counterpart_text}"
        for i, (syt_text, counterpart_text, counterpart_type) in enumerate(pairs, start=1)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pairs_text},
    ]


def judge_batch_with_llm(client, pairs: List["tuple[str, str, str]"], model: str = CHAT_MODEL,
                          instructions: str = "",
                          usage: Optional[EmbeddingUsage] = None, timeout: int = 120) -> List["tuple[str, str]"]:
    """Ask the chat model to judge a BATCH of (syt_text, counterpart_text,
    counterpart_type) pairs in ONE request, cutting the number of LLM
    calls for a run roughly by the batch size compared to
    one-request-per-pair.

    Uses the same with_raw_response pattern as the embedding calls (see
    _embed_batch_with_retry()) so exact token usage and gateway cost for
    THIS ONE batch call can be read directly and accumulated into
    ``usage``.

    Args:
        pairs: list of (syt_text, counterpart_text, counterpart_type)
               tuples, one per pair in the batch.
        instructions: judging criteria text (editable in the GUI, defaults
                      to LLM_JUDGE_DEFAULT_INSTRUCTIONS when blank).

    Returns a list of (verdict, reasoning) tuples, ALIGNED 1:1 with
    ``pairs`` by position -- if the model's response is missing an entry
    for a given pair number, or the whole call/parse fails, that pair's
    slot is filled with ("error", <reason>) rather than raising, so one
    bad batch never aborts the whole "Check Duplicates" run.
    """
    results: List["tuple[str, str]"] = [("error", "No response for this pair")] * len(pairs)
    try:
        # Some models (e.g. claude-4-7-opus-v1:0, claude-4-8-opus-v1:0 via
        # Vertex AI / litellm) reject the `temperature` parameter with
        # "temperature is deprecated for this model". Skip it for those
        # models; for all others temperature=0 gives deterministic output.
        create_kwargs: dict = dict(
            messages=_build_llm_judge_messages(pairs, instructions),
            model=model,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        # Only add temperature for models that accept it -- Claude models
        # routed through litellm/Vertex reject it outright.
        model_lower = model.lower()
        if not ("claude" in model_lower or "gemini" in model_lower):
            create_kwargs["temperature"] = 0
        raw = client.chat.completions.with_raw_response.create(**create_kwargs)
        cost = float(raw.headers.get("x-litellm-response-cost", 0) or 0)
        response = raw.parse()
        if usage is not None:
            usage.tokens += response.usage.total_tokens
            usage.cost += cost
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        for entry in parsed.get("results", []):
            pair_num = entry.get("pair")
            if not isinstance(pair_num, int) or not (1 <= pair_num <= len(pairs)):
                continue
            verdict = entry.get("verdict", "error")
            reasoning = str(entry.get("reasoning", ""))
            if verdict not in ("same_scenario", "partial_overlap", "different_scenario"):
                verdict, reasoning = "error", f"Unexpected verdict from model: {verdict!r}"
            results[pair_num - 1] = (verdict, reasoning)
    except Exception as exc:
        results = [("error", str(exc))] * len(pairs)
    return results


def llm_judgment_cache_key(syt_text: str, counterpart_text: str, counterpart_type: str,
                            model: str, instructions: str) -> str:
    """Build a content-addressed cache key for one pair's LLM judgment,
    combining the normalized texts with EVERYTHING that affects the
    verdict (counterpart kind, model, judging instructions) -- so a
    changed model or edited instructions correctly misses the cache and
    gets re-judged, while an identical pair judged under identical
    settings is never sent to the LLM twice. Hashed into a short digest
    by the caller (see trek_cache.TrekCache.get_llm_judgments()); this
    function only builds the raw composite string.
    """
    criteria = instructions.strip() if instructions and instructions.strip() else LLM_JUDGE_DEFAULT_INSTRUCTIONS
    return "|".join((
        _normalize_for_exact_match(syt_text),
        _normalize_for_exact_match(counterpart_text),
        counterpart_type, model, criteria,
    ))


def run_llm_judge_stage(client, result: DuplicateCheckResult, model: str = CHAT_MODEL,
                         instructions: str = "", batch_size: int = 1,
                         max_workers: int = 20, progress_cb=None,
                         llm_cache: Optional[Dict[str, "tuple[str, str]"]] = None,
                         cost_cache: Optional[Dict[str, float]] = None,
                         should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """Run the LLM-judge stage over every pair in ``result.pairs``, mutating
    each SimilarityResult's llm_verdict/llm_reasoning in place and
    accumulating result.total_llm_tokens / total_llm_cost_usd / llm_calls.

    This is a separate, opt-in stage layered on top of the hybrid
    BM25/vector/sequence ensemble score (see compare_syt_swt_pairs()): that
    score is fast/cheap but purely lexical/statistical, so it can be fooled
    by SYT/SWT pairs that share almost no vocabulary or structure despite
    testing the identical scenario, or vice versa. Reading both texts with
    an LLM catches cases the ensemble score alone cannot. The two results
    are kept fully SEPARATE -- llm_verdict never overrides or merges into
    the ensemble-score classification; both are reported independently.

    Pairs are grouped by SYT id first, then batched by NUMBER OF SYT
    GROUPS (``batch_size``, default 1) rather than raw pair count -- every
    pair belonging to a given SYT (its SWT pairs AND any related-SYT
    pairs, see FetchLinksWorker's related-SYT detection) always lands in
    the SAME request, never split across two. This lets the model judge
    all of one SYT's coverage together, and lets the user directly choose
    "how many SYT test cases' worth of pairs go in one request" rather
    than an opaque raw pair count that could split a SYT's own pairs
    arbitrarily. Batches are then run concurrently via a bounded
    ThreadPoolExecutor (the sync counterpart of
    group_requirements.start_grouping()'s
    asyncio.Semaphore(CONCURRENT_REQUESTS)-bounded concurrent requests --
    threads instead of asyncio tasks since this whole GUI is QThread-based,
    not asyncio-based) up to ``max_workers`` batches at a time. A shared
    client is safe here -- Bearer-token auth has no per-instance mutable
    state to race on, unlike the TREK SSPI client pattern used elsewhere
    in this codebase.

    Args:
        instructions: judging criteria text (editable in the GUI, defaults
                      to LLM_JUDGE_DEFAULT_INSTRUCTIONS when blank) applied
                      to every pair -- see _build_llm_judge_messages().
        batch_size: number of SYT groups judged per single LLM request.
        llm_cache: optional dict of llm_judgment_cache_key(...) ->
                   (verdict, reasoning), reused/updated in place (mirrors
                   compare_syt_swt_pairs()'s embedding_cache parameter) so
                   callers can persist it (e.g. to trek_cache.sqlite3)
                   across runs. A pair whose cache key is already present
                   is served for free -- no LLM call, no cost -- and
                   counted in result.llm_cached_pairs; new judgments are
                   written back into this same dict for the caller to persist.
        cost_cache: optional dict populated with {cache_key: cost_usd} for
                    every NEWLY judged pair (cache hits excluded), splitting
                    each batch's real reported cost evenly across its pairs
                    -- lets callers persist a REAL per-pair cost (e.g. into
                    trek_cache's llm_judgments table) for historical-cost
                    reporting later.
        should_cancel: optional zero-arg callable polled after each batch
                       completes -- if it returns True, any not-yet-started
                       batch is cancelled and already-running ones are not
                       waited on, so the caller can implement a responsive
                       "Stop" button without corrupting result (whatever
                       was judged before cancellation is noticed is kept).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not result.pairs:
        return

    result.llm_model = model

    # Stage 0: serve whatever's already cached for free (identical pair,
    # same model, same instructions), and only queue the rest for the LLM.
    # IMPORTANT: cached ERROR verdicts (e.g. "overbudget", network failure,
    # expired token) are treated as cache MISSES -- a transient failure
    # should never be permanently sticky, so the pair is re-queued for a
    # fresh LLM call instead of replaying the old error indefinitely.
    to_judge: List[SimilarityResult] = []
    for p in result.pairs:
        cache_key = llm_judgment_cache_key(p.syt_text, p.swt_text, p.counterpart_type, model, instructions)
        cached = llm_cache.get(cache_key) if llm_cache is not None else None
        if cached is not None and cached[0] != "error":
            p.llm_verdict, p.llm_reasoning = cached
            if cost_cache is not None and cache_key in cost_cache:
                p.llm_cost_usd = cost_cache[cache_key]
            result.llm_cached_pairs += 1
        else:
            to_judge.append(p)

    if not to_judge:
        return

    # Group the remaining (not-cached) pairs by SYT id, preserving
    # first-seen order, so a batch is always a whole number of complete
    # SYT groups.
    groups: Dict[str, List[SimilarityResult]] = {}
    for p in to_judge:
        groups.setdefault(p.syt_id, []).append(p)
    syt_ids_ordered = list(groups.keys())

    batch_size = max(1, batch_size)
    syt_id_batches = [syt_ids_ordered[i:i + batch_size] for i in range(0, len(syt_ids_ordered), batch_size)]
    batches = [[p for sid in sid_group for p in groups[sid]] for sid_group in syt_id_batches]
    total_batches = len(batches)
    total_pairs = len(to_judge)

    def _judge_batch(batch_pairs: List[SimilarityResult]):
        batch_usage = EmbeddingUsage()
        verdicts = judge_batch_with_llm(
            client, [(p.syt_text, p.swt_text, p.counterpart_type) for p in batch_pairs],
            model=model, instructions=instructions, usage=batch_usage,
        )
        return batch_pairs, verdicts, batch_usage

    done_pairs = 0
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, total_batches))) as executor:
        futures = [executor.submit(_judge_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_pairs, verdicts, batch_usage = future.result()
            cost_per_pair = batch_usage.cost / len(batch_pairs) if batch_pairs else 0.0
            for pair, (verdict, reasoning) in zip(batch_pairs, verdicts):
                pair.llm_verdict = verdict
                pair.llm_reasoning = reasoning
                pair.llm_cost_usd = cost_per_pair
                # Only cache REAL verdicts -- never persist "error" results
                # (transient failures like overbudget/network/expired token)
                # so they don't poison the cache and replay indefinitely on
                # subsequent runs even after the user fixes the root cause.
                if llm_cache is not None and verdict != "error":
                    cache_key = llm_judgment_cache_key(
                        pair.syt_text, pair.swt_text, pair.counterpart_type, model, instructions
                    )
                    llm_cache[cache_key] = (verdict, reasoning)
                    if cost_cache is not None:
                        cost_cache[cache_key] = cost_per_pair
            result.total_llm_tokens += batch_usage.tokens
            result.total_llm_cost_usd += batch_usage.cost
            result.llm_calls += 1
            done_pairs += len(batch_pairs)
            if progress_cb:
                progress_cb(f"LLM verification: {done_pairs}/{total_pairs} new pair(s) judged "
                             f"({result.llm_cached_pairs} from cache, "
                             f"{result.llm_calls}/{total_batches} batch requests)...")
            if should_cancel is not None and should_cancel():
                executor.shutdown(wait=False, cancel_futures=True)
                break

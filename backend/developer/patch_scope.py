"""
Deterministic guard against additive requests producing destructive
patches.

Root cause this exists for: developer_node's exact-snippet patch prompt
(backend/graph/nodes.py) shows small non-Python files (README.md, config,
etc.) to the LLM as a single "[COMPLETE FILE CONTENT]" block and tells it
that an empty `original_code_snippet` means "replace the entire file" -
convenient for genuine rewrites, but nothing stopped the model from also
reaching for that convention on a purely additive request ("add an E2E Test
section"), regenerating the whole document and incidentally dropping most
of its original content. Policy's deletion-ratio risk check
(backend/vcs/git_manager.py) catches this too, but only late (post-hoc risk
scoring for HITL review) and without knowing whether the user actually
asked for a rewrite. This module is evaluated deterministically from the
actual request text and diff - never an LLM judgment - so it can reject the
candidate patch before it ever reaches approval.
"""

import difflib
import re
from typing import Optional

# Deliberately simple, explicit phrase matching - not an LLM call and not a
# general-purpose NLP classifier. False negatives here just mean the guard
# doesn't fire (falling back to the existing policy/QA checks); false
# positives are bounded by requiring an explicit rewrite phrase to always
# win (see is_explicit_rewrite_request / detect_unsafe_additive_rewrite).
_ADDITIVE_PATTERNS = [
    r"\badd\b",
    r"\bappend\b",
    r"\binsert\b",
    r"\bnew section\b",
    r"\bextend\b",
]

_REWRITE_PATTERNS = [
    r"\brewrite\b",
    r"\breplace (the )?(content|contents|file|document)\b",
    r"\brestructure\b",
    r"\bregenerate\b",
    r"\boverwrite\b",
    r"\bstart (over|from scratch)\b",
    # Plural "sections" only (matches the literal requirement wording
    # "remove specified sections") - a sweeping, multi-section removal is
    # a genuine rewrite-scale request. Singular "section" ("remove the
    # obsolete section") is a single, targeted removal and must NOT bypass
    # the deletion-fraction check below: a mixed request like "add an E2E
    # Test section and remove the obsolete section" only authorizes
    # removing one section, not replacing most of the file - confirmed
    # against a real case where that phrasing was used to smuggle a
    # near-total rewrite past this guard.
    r"\bremove\b[^.\n]*\bsections\b",
    r"\bdelete\b[^.\n]*\bsections\b",
]

# Fraction of the ORIGINAL file's lines a patch may delete before an
# additive, non-rewrite request is treated as a patch-generation failure.
# Matches the deletion-ratio threshold GitWorkspaceManager.evaluate_risk
# already uses for its own (differently-scoped) risk heuristic.
DEFAULT_DELETION_FRACTION_THRESHOLD = 0.5


def is_additive_request(user_request: str) -> bool:
    """True when the user's request reads as adding/appending/inserting
    content rather than rewriting or replacing it."""
    text = (user_request or "").lower()
    return any(re.search(p, text) for p in _ADDITIVE_PATTERNS)



# Negation cues that flip a rewrite keyword's meaning when they appear in
# the same clause immediately before it - "do not rewrite" is an additive-
# preservation instruction, not a rewrite request, even though it contains
# the word "rewrite". Deliberately simple phrase matching, matching the
# rest of this module's approach: a false negative here just means the
# negation isn't recognized and the (safe) rewrite-request classification
# stands; there is no false-positive risk since this only ever suppresses
# a match, never creates one.
_NEGATION_CUES = [
    r"\bdo not\b", r"\bdon't\b", r"\bdont\b",
    r"\bdoes not\b", r"\bdoesn't\b",
    r"\bshould not\b", r"\bshouldn't\b",
    r"\bmust not\b", r"\bmustn't\b",
    r"\bcannot\b", r"\bcan't\b", r"\bcan not\b",
    r"\bnever\b", r"\bwithout\b", r"\bavoid\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_CUES))
_CLAUSE_BOUNDARY_RE = re.compile(r"[.\n;]")


def _is_negated_at(text: str, match_start: int) -> bool:
    """True when a negation cue (see _NEGATION_CUES) appears earlier in the
    SAME clause as the match at `match_start` - i.e. after the previous
    sentence/clause boundary (., newline, or ;) and before the match
    itself. A negation in an earlier, unrelated sentence never suppresses
    a later, genuine rewrite request."""
    boundaries = [m.end() for m in _CLAUSE_BOUNDARY_RE.finditer(text, 0, match_start)]
    clause_start = boundaries[-1] if boundaries else 0
    clause = text[clause_start:match_start]
    return bool(_NEGATION_RE.search(clause))


def is_explicit_rewrite_request(user_request: str) -> bool:
    """True when the user explicitly asked to rewrite, replace, restructure,
    regenerate, or remove sections from the document - but NOT when that
    same instruction is negated in the same clause (e.g. "do not rewrite
    the existing content"), which is an additive-preservation instruction,
    not a rewrite request."""
    text = (user_request or "").lower()
    for pattern in _REWRITE_PATTERNS:
        for match in re.finditer(pattern, text):
            if not _is_negated_at(text, match.start()):
                return True
    return False


def compute_deletion_fraction(original_content: str, updated_content: str) -> float:
    """
    Fraction of the ORIGINAL file's lines a patch deletes, via a line-level
    diff. Deliberately different from GitWorkspaceManager.evaluate_risk's
    deletion_ratio (lines_deleted / (lines_added + lines_deleted), i.e. "how
    much of this diff is deletion") - this measures "how much of the
    starting document survives", which is what distinguishes an additive
    edit from a disguised rewrite regardless of how much new content was
    also added alongside the deletions.
    """
    original_lines = original_content.splitlines()
    if not original_lines:
        return 0.0
    updated_lines = updated_content.splitlines()
    diff = difflib.unified_diff(original_lines, updated_lines, lineterm="")
    deleted = sum(
        1 for line in diff if line.startswith("-") and not line.startswith("---")
    )
    return deleted / len(original_lines)


def detect_unsafe_additive_rewrite(
    user_request: str,
    original_content: str,
    updated_content: str,
    threshold: float = DEFAULT_DELETION_FRACTION_THRESHOLD,
) -> Optional[str]:
    """
    Returns a human-readable reason when a patch looks like it replaced
    most of an existing file in response to a request that only asked to
    add something. Returns None when the patch is safe to proceed: the
    request isn't additive, the user explicitly asked for a rewrite, or the
    deletion is within the safety threshold.
    """
    if is_explicit_rewrite_request(user_request):
        return None
    if not is_additive_request(user_request):
        return None

    fraction = compute_deletion_fraction(original_content, updated_content)
    if fraction <= threshold:
        return None

    return (
        f"Request appears additive but the patch deletes {fraction:.0%} of "
        f"the original file's content, exceeding the {threshold:.0%} safety "
        f"threshold. Additive requests must preserve unrelated existing "
        f"content unless a rewrite is explicitly requested."
    )


# Literal patch-editor/diff-tool wrapper syntax (the OpenAI "apply_patch"
# tool format, and close variants) that an LLM can occasionally hallucinate
# INSTEAD OF the actual file content it was asked to produce - confirmed in
# production (run_7fb6d95b60d9): the model returned
#     *** Begin Patch
#     *** Update File: README.md
#     @@
#     ...
#     *** End Patch
# as updated_code_snippet, and this got written verbatim into README.md as
# if it were the file's real content. Deliberately checked unconditionally
# (unlike detect_unsafe_additive_rewrite above) - this is never legitimate
# file content for ANY request, additive or an explicit rewrite alike, so
# it must never be gated behind the additive/rewrite classification.
_PATCH_WRAPPER_MARKERS_RE = re.compile(
    r"^\*\*\* (Begin Patch|End Patch|(Update|Add|Delete) File:)",
    re.MULTILINE,
)


def detect_patch_wrapper_artifacts(content: str) -> Optional[str]:
    """
    Returns a human-readable reason when `content` contains literal
    patch-editor/diff-tool wrapper markers (e.g. "*** Begin Patch", "***
    Update File: ...", "*** End Patch") instead of genuine file content.
    Returns None when no such marker is found.
    """
    match = _PATCH_WRAPPER_MARKERS_RE.search(content or "")
    if not match:
        return None
    return (
        f"Generated content contains literal patch-editor wrapper syntax "
        f"({match.group(0)!r}) instead of the file's actual content - this "
        f"looks like a hallucinated patch-tool format, not real content."
    )


class PatchWrapperArtifactError(ValueError):
    """
    Raised when generated content that was about to be written to disk
    contains literal patch-editor wrapper syntax (see
    detect_patch_wrapper_artifacts) instead of real file content. Callers
    (developer_node/_materialize_developer_changes) catch this specifically
    to route the failure through the existing bounded revision loop -
    exactly like a SafePatcher anchor mismatch - rather than writing the
    wrapper text to disk or crashing the run.
    """

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"{file_path}: {reason}")

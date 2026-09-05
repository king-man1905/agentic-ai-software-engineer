from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.indexer.models import CodeChunk
from backend.schemas.rag import (
    RetrievalEvaluation,
    RetrievalEvaluationStatus,
    RetrievedDocumentView,
)

MAX_RETRIEVAL_ATTEMPTS = 2


def extract_potential_filenames(text: str) -> List[str]:
    """Extracts file names or paths mentioned in text (e.g., auth/service.py, test_token.py)."""
    pattern = r"\b[a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml|md|txt|html|css)\b"
    matches = re.findall(pattern, text)
    return list(dict.fromkeys(matches))


def extract_potential_symbols(text: str) -> List[str]:
    """Extracts function, method, or class identifiers mentioned in text."""
    # Matches:
    # 1. Qualified names: e.g., AuthService.validate_token, os.path.join
    # 2. PascalCase / CamelCase with multiple bumps: e.g., AuthService, TokenManager, ConnectionPool
    # 3. snake_case with underscores: e.g., validate_token, calculate_tax
    qualified_pattern = r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
    camel_pattern = r"\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*\b"
    snake_pattern = r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"

    raw_matches = (
        re.findall(qualified_pattern, text)
        + re.findall(camel_pattern, text)
        + re.findall(snake_pattern, text)
    )
    skip_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".md", ".txt", ".html", ".css"}
    common_words = {
        "github", "issue", "python", "test", "run", "fastapi", "agent", "true",
        "false", "none", "null", "fix", "update", "patch", "error", "warning"
    }

    filtered = []
    for m in dict.fromkeys(raw_matches):
        if any(m.endswith(ext) for ext in skip_extensions):
            continue
        if m.lower() in common_words:
            continue
        filtered.append(m)

    return filtered


def extract_error_messages(text: str) -> List[str]:
    """Extracts error traces or exception keywords from text."""
    pattern = r"\b[A-Za-z]+(?:Error|Exception|Warning|Fault)\b(?::[^\n]+)?"
    matches = re.findall(pattern, text)
    return list(dict.fromkeys(matches))


class RetrievalEvaluator:
    """
    Deterministic evaluation layer that checks if retrieved repository context
    is sufficient for the requested engineering task before code synthesis.
    """

    @classmethod
    def evaluate(
        cls,
        issue_text: str,
        query: str,
        chunks: List[CodeChunk],
        scores: Optional[List[float]] = None,
    ) -> RetrievalEvaluation:
        """
        Evaluates the relevance and sufficiency of retrieved CodeChunks.
        """
        if not chunks:
            return RetrievalEvaluation(
                status=RetrievalEvaluationStatus.INSUFFICIENT.value,
                confidence=0.0,
                relevant_documents=[],
                missing_context=["No matching repository documents found."],
                reason="Retrieval yielded zero candidate documents from the repository index.",
            )

        target_files = extract_potential_filenames(issue_text)
        target_symbols = extract_potential_symbols(issue_text)
        target_errors = extract_error_messages(issue_text)

        retrieved_views: List[RetrievedDocumentView] = []
        for i, c in enumerate(chunks):
            score = scores[i] if scores and i < len(scores) else 1.0
            retrieved_views.append(
                RetrievedDocumentView(
                    file=c.file_path,
                    symbol=c.symbol_name,
                    line_start=c.start_line,
                    line_end=c.end_line,
                    content=c.content,
                    score=round(score, 4),
                    symbol_type=c.symbol_type,
                    source_hash=c.source_hash,
                )
            )

        retrieved_files_set = {c.file_path.replace("\\", "/").lower() for c in chunks}
        retrieved_symbols_set = {
            c.symbol_name.lower() for c in chunks if c.symbol_name
        }
        retrieved_content_blob = " ".join(c.content.lower() for c in chunks)

        missing_context: List[str] = []
        matched_files = []
        matched_symbols = []

        # 1. Check targeted file references
        for tf in target_files:
            clean_tf = tf.replace("\\", "/").lower()
            matched = any(
                clean_tf in rf or rf.endswith(clean_tf) or Path(rf).name == Path(clean_tf).name
                for rf in retrieved_files_set
            )
            if matched:
                matched_files.append(tf)
            else:
                missing_context.append(f"Referenced file not retrieved: {tf}")

        # 2. Check targeted symbol references
        for ts in target_symbols:
            clean_ts = ts.lower()
            ts_parts = [p.lower() for p in ts.split(".") if p]
            matched = any(
                clean_ts == rs
                or clean_ts in rs.split(".")
                or rs in ts_parts
                for rs in retrieved_symbols_set
            ) or clean_ts in retrieved_content_blob
            if matched:
                matched_symbols.append(ts)
            else:
                missing_context.append(f"Referenced symbol not retrieved: {ts}")

        # 3. Check error traces
        matched_errors = []
        for err in target_errors:
            clean_err = err.split(":")[0].lower()
            if clean_err in retrieved_content_blob:
                matched_errors.append(err)
            else:
                missing_context.append(f"Error context not found: {err}")

        # ---------------------------------------------------------------------
        # Scoring & Sufficiency Decision
        # ---------------------------------------------------------------------
        total_targets = len(target_files) + len(target_symbols)
        matched_targets = len(matched_files) + len(matched_symbols)

        # Case A: Explicit targets mentioned in issue, but none were retrieved
        if total_targets > 0 and matched_targets == 0:
            return RetrievalEvaluation(
                status=RetrievalEvaluationStatus.INSUFFICIENT.value,
                confidence=0.25,
                relevant_documents=[],
                missing_context=missing_context,
                reason=(
                    f"Identified targets ({', '.join(target_files + target_symbols)}) "
                    f"were completely absent from retrieved repository context."
                ),
            )

        # Case B: Some targets retrieved, but others missing
        if total_targets > 0 and matched_targets < total_targets:
            confidence = round(0.5 + (0.3 * (matched_targets / total_targets)), 2)
            relevant_docs = [
                v for v in retrieved_views
                if any(mf.lower() in v.file.lower() for mf in matched_files)
                or (v.symbol and any(ms.lower() in v.symbol.lower() for ms in matched_symbols))
            ] or retrieved_views[:2]

            return RetrievalEvaluation(
                status=RetrievalEvaluationStatus.PARTIAL.value,
                confidence=confidence,
                relevant_documents=relevant_docs,
                missing_context=missing_context,
                reason=(
                    f"Partial match: Retrieved {matched_targets}/{total_targets} identified targets. "
                    f"Missing: {', '.join(missing_context[:3])}."
                ),
            )

        # Case C: Free-text issue without explicit filenames/symbols
        if total_targets == 0:
            # Check keyword relevance across tokens
            query_tokens = [w.lower() for w in re.findall(r"\w{4,}", query)]
            overlap_count = sum(1 for t in query_tokens if t in retrieved_content_blob)
            ratio = overlap_count / max(len(query_tokens), 1)

            if ratio >= 0.4:
                return RetrievalEvaluation(
                    status=RetrievalEvaluationStatus.RELEVANT.value,
                    confidence=min(0.95, round(0.75 + (ratio * 0.2), 2)),
                    relevant_documents=retrieved_views,
                    missing_context=[],
                    reason="Retrieved documents provide strong semantic and term coverage for the request.",
                )
            elif ratio >= 0.15:
                return RetrievalEvaluation(
                    status=RetrievalEvaluationStatus.PARTIAL.value,
                    confidence=0.60,
                    relevant_documents=retrieved_views[:2],
                    missing_context=["Broad keyword coverage is low."],
                    reason="Moderate relevance; context may lack full component implementation details.",
                )
            else:
                return RetrievalEvaluation(
                    status=RetrievalEvaluationStatus.INSUFFICIENT.value,
                    confidence=0.30,
                    relevant_documents=[],
                    missing_context=["No overlapping component code found."],
                    reason="Retrieved documents share minimal terminology with the request.",
                )

        # Case D: All identified targets matched
        return RetrievalEvaluation(
            status=RetrievalEvaluationStatus.RELEVANT.value,
            confidence=0.92,
            relevant_documents=retrieved_views,
            missing_context=[],
            reason=f"All identified repository targets ({', '.join(matched_files + matched_symbols)}) were successfully retrieved.",
        )


class QueryRewriter:
    """
    Synthesizes targeted alternative queries when initial retrieval is INSUFFICIENT or PARTIAL.
    Enforces a strict upper bound of MAX_RETRIEVAL_ATTEMPTS.
    """

    @classmethod
    def rewrite(
        cls,
        issue_text: str,
        current_query: str,
        missing_context: Optional[List[str]] = None,
        attempt: int = 1,
    ) -> Optional[str]:
        """
        Constructs a refined search query using entities, symbol names, and missing context.
        Returns None if max refinement attempts have been reached.
        """
        if attempt >= MAX_RETRIEVAL_ATTEMPTS:
            return None

        terms: List[str] = []

        # 1. Prioritize explicit files or symbols mentioned
        files = extract_potential_filenames(issue_text)
        symbols = extract_potential_symbols(issue_text)
        errors = extract_error_messages(issue_text)

        for f in files:
            # Add stem and name
            p = Path(f)
            terms.append(p.stem)
            terms.append(p.name)

        for s in symbols:
            terms.append(s)
            if "." in s:
                terms.extend(s.split("."))

        for e in errors:
            terms.append(e.split(":")[0])

        # 2. Extract technical action verbs and nouns
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", issue_text)
        skip = {"please", "would", "could", "should", "using", "about", "their", "where", "which", "there"}
        tech_words = [w for w in keywords if w.lower() not in skip and w not in terms]

        terms.extend(tech_words[:6])

        # Deduplicate while maintaining order
        deduped = list(dict.fromkeys(terms))
        if not deduped:
            return f"{current_query} implementation"

        return " ".join(deduped[:8])

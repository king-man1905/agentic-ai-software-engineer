import fnmatch
import os
import posixpath
import re
from typing import List, Optional, Tuple


def is_traversal_attack(path_str: str) -> bool:
    """
    Checks whether a path contains path traversal tokens (e.g. '../', '..\\').
    """
    cleaned = path_str.replace("\\", "/")
    return (
        "../" in cleaned
        or cleaned.startswith("..")
        or "/.." in cleaned
        or cleaned.endswith("/..")
    )


def normalize_path(path_str: str) -> str:
    """
    Canonicalizes a file path for safe policy inspection.
    Strips absolute drives, normalizes slashes, and resolves dot-segments.
    """
    cleaned = path_str.replace("\\", "/").strip()

    # Strip drive letter if present (e.g. C:/)
    if re.match(r"^[a-zA-Z]:", cleaned):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")

    # Normalize dot-segments safely using posixpath
    normalized = posixpath.normpath(cleaned)

    # Strip any remaining leading '../' or './'
    while normalized.startswith("../") or normalized.startswith("./"):
        normalized = normalized[3:] if normalized.startswith("../") else normalized[2:]

    if normalized in ("..", "."):
        normalized = ""

    return normalized


def matches_protected_path(
    file_path: str,
    protected_patterns: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a given file path matches any protected path pattern.
    Returns (is_matched, matching_pattern).
    """
    norm_path = normalize_path(file_path)
    basename = posixpath.basename(norm_path)

    for pattern in protected_patterns:
        norm_pattern = pattern.replace("\\", "/").strip().lstrip("/")

        # 1. Directory recursive prefix match (e.g., auth/** or secrets/**)
        if norm_pattern.endswith("/**"):
            prefix = norm_pattern[:-3]
            if norm_path == prefix or norm_path.startswith(prefix + "/"):
                return True, pattern

        # 2. Exact match
        if norm_path == norm_pattern:
            return True, pattern

        # 3. Full path fnmatch
        if fnmatch.fnmatch(norm_path, norm_pattern):
            return True, pattern

        # 4. Basename fnmatch (e.g., .env* matching config/.env or src/.env.local)
        if fnmatch.fnmatch(basename, norm_pattern):
            return True, pattern

    return False, None

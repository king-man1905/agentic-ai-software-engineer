"""
Tenant-namespaced workspace path resolution.

Every workspace a run operates against lives at:

    workspace/<organization_id>/<project_id>

instead of the previous, unnamespaced `workspace/<project_id>`. That older
form let two different tenants collide on the same directory merely by
choosing the same project_id (project_id is a caller-supplied string with
no server-side uniqueness enforcement) - `resolve_workspace_path()` is the
single place every caller must go through instead of constructing
`Path("workspace") / project_id` (or any namespaced variant of it)
independently, so this property can't silently regress at a fifth call
site the way it drifted across five before this fix.
"""

import os
from pathlib import Path
from typing import Optional

from backend.policy.path_filter import is_traversal_attack


def safe_path_component(value: Optional[str]) -> Optional[str]:
    """
    Validates that `value` is safe to use as a single path *segment*
    (organization_id or project_id) - not a sub-path. Rejects anything
    empty, containing a path separator (so a caller can never smuggle
    extra segments, e.g. "org/../../other-org", through what's supposed
    to be one component), a bare '.'/'..', or a Windows drive-letter
    prefix. Returns the stripped value, or None if unsafe.

    Shared across every tenant-namespaced path resolver (workspace/ and
    vector_store/) so this validation can't drift between them.
    """
    candidate = (value or "").strip()
    if not candidate:
        return None
    if "/" in candidate or "\\" in candidate:
        return None
    if candidate in (".", ".."):
        return None
    if is_traversal_attack(candidate):
        return None
    if len(candidate) >= 2 and candidate[1] == ":" and candidate[0].isalpha():
        return None
    return candidate


def resolve_workspace_path(organization_id: Optional[str], project_id: Optional[str]) -> Optional[Path]:
    """
    Resolves the tenant-namespaced workspace directory for
    (organization_id, project_id). Returns None if either component is
    unsafe - callers MUST fail closed on None rather than falling back to
    an unnamespaced or guessed path.

    Preserves the existing dual-resolution behavior every call site used
    to hand-roll: prefers the path relative to the current working
    directory if it already exists there (as it does inside most test
    fixtures, which monkeypatch cwd rather than os.getcwd()); otherwise
    resolves explicitly via os.getcwd() (as production code, which
    monkeypatches os.getcwd() in some tests, needs).
    """
    org = safe_path_component(organization_id)
    proj = safe_path_component(project_id)
    if org is None or proj is None:
        return None

    relative = Path("workspace") / org / proj
    if relative.exists():
        return relative
    return Path(os.getcwd()) / "workspace" / org / proj

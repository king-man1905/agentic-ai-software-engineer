"""
Persistent storage for repository registry metadata.

Root cause this exists for: TenantManager's repository registry
(backend/security/tenant.py) was in-memory only - a FastAPI/Uvicorn
process restart silently forgot every registered repository. The next run
against that repository then failed workspace provisioning with
WORKSPACE_PROVISIONING_FAILED ("repository ... is not authorized for
organization ...") - correctly fail-closed given what TenantManager
actually knew at that point, but the real problem was that the
authorization itself had evaporated on restart, not a genuine
authorization decision.

Repository identity is tenant-scoped: a registration is keyed by
(organization_id, id), never by id/full_name alone. An earlier version of
this store used `id TEXT PRIMARY KEY` - a single, globally-shared key -
which meant a second organization registering the same repository id/
full_name (e.g. two tenants both legitimately registering the public repo
"acme/widgets") would silently overwrite the first organization's row,
corrupting its registration. get()/upsert() are both tenant-scoped and
require organization_id; there is no method here that performs an
unscoped, cross-tenant lookup - see TenantManager.find_registration_any_organization
for the one legitimate cross-tenant use (registration-conflict detection),
which is served from the in-memory registry, never from this store
directly.

Stores only non-secret repository metadata (id, organization_id, name,
full_name, default_branch, allowed_branches, is_private, is_authorized,
timestamps). The GitHub token is NEVER read or written here - it
continues to live only in the in-memory Repository object for the
lifetime of the process that received it via POST /api/v1/repositories.
A repository rehydrated from this store after a restart has
github_token=None, so the existing `repo.github_token or
os.environ.get("GITHUB_TOKEN")` fallback (backend/graph/runner.py,
backend/api/app.py) transparently falls back to the environment
variable, exactly as it already does for any repository registered
without an explicit token.

Deliberately its own database file (workspace/repositories.db by
default), never workspace/telemetry.db - a repository registry is not
telemetry data, and keeping it separate means this table can never
collide with, or risk migrating, telemetry.db's existing schema/data on
an existing deployment. Mirrors backend/observability/store.py's
TelemetryStore conventions: WAL mode, a re-entrant lock, and
CREATE TABLE IF NOT EXISTS so the application starts successfully
whether or not the table (or the whole file) exists yet.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.schemas.tenant import Repository

_COMPOSITE_PK_MARKER = "PRIMARY KEY (organization_id, id)"


class RepositoryStore:
    """Thread-safe SQLite-backed persistence for repository registry metadata."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Resolved to an absolute path at construction time (not left
            # as a relative "workspace/..." string) so this long-lived
            # singleton's on-disk location stays fixed even if the
            # process's current working directory later changes (e.g.
            # tests that monkeypatch.chdir(tmp_path) to redirect
            # workspace-relative repo clones - those must never also
            # silently redirect this registry to a fresh, uninitialized
            # SQLite file with no tables).
            workspace_dir = Path("workspace").resolve()
            workspace_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(workspace_dir / "repositories.db")
        else:
            self.db_path = str(Path(db_path).resolve())

        self._lock = threading.RLock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            self._migrate_legacy_global_key_schema(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_name TEXT,
                    default_branch TEXT NOT NULL DEFAULT 'main',
                    allowed_branches TEXT NOT NULL,
                    is_private INTEGER NOT NULL DEFAULT 1,
                    is_authorized INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repositories_full_name ON repositories (organization_id, full_name);")
            conn.commit()

    def _migrate_legacy_global_key_schema(self, conn: sqlite3.Connection) -> None:
        """
        Rebuilds a pre-existing `repositories` table created by the
        original version of this store, whose PRIMARY KEY was `id` alone -
        a single global namespace where a second organization registering
        the same id/full_name silently overwrote the first organization's
        row. Every legacy row already has its own organization_id column
        (just not as part of the key), so the existing rows are preserved
        as-is under the new composite key; only rows that had ALREADY been
        overwritten before this migration ever ran are unrecoverable
        (that data loss happened under the old schema, not something a
        migration run afterward can undo). A no-op for a fresh install
        (table doesn't exist yet) or a database already on the current
        schema.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='repositories';"
        ).fetchone()
        if row is None or row[0] is None:
            return
        if _COMPOSITE_PK_MARKER in row[0]:
            return

        conn.execute("ALTER TABLE repositories RENAME TO repositories_legacy_single_key;")
        conn.execute(
            """
            CREATE TABLE repositories (
                id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                name TEXT NOT NULL,
                full_name TEXT,
                default_branch TEXT NOT NULL DEFAULT 'main',
                allowed_branches TEXT NOT NULL,
                is_private INTEGER NOT NULL DEFAULT 1,
                is_authorized INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, id)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO repositories (
                id, organization_id, name, full_name, default_branch,
                allowed_branches, is_private, is_authorized, created_at, updated_at
            )
            SELECT id, organization_id, name, full_name, default_branch,
                   allowed_branches, is_private, is_authorized, created_at, updated_at
            FROM repositories_legacy_single_key;
            """
        )
        conn.execute("DROP TABLE repositories_legacy_single_key;")
        conn.commit()

    def upsert(self, repo: Repository) -> None:
        """
        Persists non-secret repository metadata only, scoped to
        repo.organization_id. repo.github_token is deliberately never
        read here - never even accessed, let alone written, so a raw
        token can never reach this table by accident.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            existing = conn.execute(
                "SELECT created_at FROM repositories WHERE organization_id = ? AND id = ?;",
                (repo.organization_id, repo.id),
            ).fetchone()
            created_at = existing["created_at"] if existing else now_iso
            conn.execute(
                """
                INSERT INTO repositories (
                    id, organization_id, name, full_name, default_branch,
                    allowed_branches, is_private, is_authorized, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organization_id, id) DO UPDATE SET
                    name=excluded.name,
                    full_name=excluded.full_name,
                    default_branch=excluded.default_branch,
                    allowed_branches=excluded.allowed_branches,
                    is_private=excluded.is_private,
                    is_authorized=excluded.is_authorized,
                    updated_at=excluded.updated_at;
                """,
                (
                    repo.id, repo.organization_id, repo.name, repo.full_name,
                    repo.default_branch, json.dumps(repo.allowed_branches),
                    1 if repo.is_private else 0, 1 if repo.is_authorized else 0,
                    created_at, now_iso,
                ),
            )
            conn.commit()

    def get(self, repo_id: str, organization_id: str) -> Optional[Repository]:
        """
        Tenant-scoped lookup by id OR full_name, within organization_id
        only - matching TenantManager's own dual-keyed in-memory lookup
        (register_repository indexes by both when they differ).
        organization_id is required and never defaulted: this store never
        performs an unscoped, cross-tenant lookup. A falsy organization_id
        fails closed (returns None) rather than searching globally.
        github_token is always None on the returned Repository - it was
        never persisted.
        """
        if not organization_id:
            return None
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM repositories WHERE organization_id = ? AND (id = ? OR full_name = ?) LIMIT 1;",
                (organization_id, repo_id, repo_id),
            ).fetchone()
            return self._row_to_repository(row) if row else None

    def list_all(self) -> List[Repository]:
        """Every persisted repository, across all tenants - used only to
        rehydrate TenantManager's in-memory registry at startup, which
        re-applies the same (organization_id, id) tenant scoping to each
        row it loads. Never exposed directly through the API, which
        always filters by tenant."""
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM repositories;")
            return [self._row_to_repository(row) for row in cursor.fetchall()]

    def reset(self) -> None:
        """Clears all records for clean test isolation."""
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM repositories;")
            conn.commit()

    @staticmethod
    def _row_to_repository(row: sqlite3.Row) -> Repository:
        return Repository(
            id=row["id"],
            organization_id=row["organization_id"],
            name=row["name"],
            full_name=row["full_name"],
            default_branch=row["default_branch"],
            allowed_branches=json.loads(row["allowed_branches"]),
            is_private=bool(row["is_private"]),
            is_authorized=bool(row["is_authorized"]),
            github_token=None,
        )

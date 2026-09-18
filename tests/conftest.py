"""
Shared pytest fixtures.
"""
import os
import subprocess

import pytest


@pytest.fixture
def fake_clone_creates_real_git_repo():
    """
    Stand-in for GitWorkspaceManager.clone_repository() used by tests that
    mock cloning: creates a REAL, minimal git repository at project_path
    with its own `.git` and an `origin` remote set to the exact clone_url
    it was asked to clone - satisfying AgentRunner._ensure_workspace_provisioned's
    post-clone verification (a workspace must demonstrably be its own valid
    git clone with a matching origin, never just "some directory exists"),
    the same way a real `git clone <clone_url> <project_path>` would.
    """
    def _fake_clone(clone_url, project_path, timeout=60, auth_header=None):
        os.makedirs(project_path, exist_ok=True)
        subprocess.run(["git", "init"], cwd=project_path, capture_output=True, text=True, check=True)
        # A directory being "(re-)cloned" may already have a .git with a
        # stale origin (e.g. a different repository's leftover clone) -
        # set-url-or-add mirrors what a real `git clone` into a fresh
        # checkout would leave behind either way.
        result = subprocess.run(
            ["git", "remote", "set-url", "origin", clone_url],
            cwd=project_path, capture_output=True, text=True,
        )
        if result.returncode != 0:
            subprocess.run(
                ["git", "remote", "add", "origin", clone_url],
                cwd=project_path, capture_output=True, text=True, check=True,
            )
        return True
    return _fake_clone

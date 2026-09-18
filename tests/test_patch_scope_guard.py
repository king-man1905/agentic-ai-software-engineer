"""
Regression tests for the additive-request/destructive-patch bug:

Real E2E bug: "Add an 'E2E Test' section to README.md..." against a
134-line README produced a patch with +25/-120 - the LLM replaced almost
the entire file instead of inserting one section. QA (LLM-judged) reported
PASS; only policy's post-hoc deletion-ratio risk check caught it, and only
because it happened to cross that unrelated threshold.

Covers:
1. A 134-line README + additive request, patched via a proper minimal
   exact-snippet insert, preserves all unrelated content.
2. The deterministic patch_scope guard rejects an additive request whose
   patch deletes most of the original file - before approval, independent
   of any LLM verdict.
3. An explicit rewrite request is allowed to delete most of the file.
4. A normal small edit is allowed.
5. SafePatcher's exact-match behavior is untouched by this guard.
6. The original_code_snippet="" whole-file-replacement convention is
   untouched (used correctly by the guard for the rewrite case).
"""

from backend.developer.models import FilePatch
from backend.developer.patch_scope import (
    compute_deletion_fraction,
    detect_unsafe_additive_rewrite,
    is_additive_request,
    is_explicit_rewrite_request,
)
from backend.developer.patcher import SafePatcher
from backend.qa.judge import StructuredQAJudge
from backend.qa.pipeline import QualityPipeline
from backend.schemas.qa import FailureCategory, QualityCheckStatus


def _readme_134_lines() -> str:
    """A README shaped like the real bug report: many unrelated sections,
    134 total lines."""
    lines = ["# Agentic AI Software Engineer", ""]
    section = 1
    while len(lines) < 134:
        lines.extend(
            [
                f"## Section {section}",
                f"This is paragraph content for section {section}.",
                f"It explains detail {section} of the pipeline.",
                "",
            ]
        )
        section += 1
    lines = lines[:134]
    content = "\n".join(lines) + "\n"
    assert len(content.splitlines()) == 134, len(content.splitlines())
    return content


ADDITIVE_REQUEST = (
    'Add an "E2E Test" section to README.md. Explain that this section is '
    "used to verify the autonomous software-engineering pipeline and "
    "GitHub pull-request workflow."
)


class TestAdditiveMinimalPatchPreservesContent:
    """1. Existing 134-line README + additive request -> generated patch
    must preserve unrelated README content."""

    def test_proper_minimal_insert_preserves_everything_else(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        # The anchor a well-behaved patch would use: a short, exact quote
        # of the file's last few lines, with the new section appended.
        new_section = (
            "\n## E2E Test\n\n"
            "This section verifies the autonomous software-engineering "
            "pipeline and GitHub pull-request workflow.\n"
        )
        anchor = "\n".join(original.splitlines()[-3:])
        patch = FilePatch(
            file_path="README.md",
            original_code_snippet=anchor,
            updated_code_snippet=anchor + new_section,
            explanation="Add E2E Test section",
        )

        result = SafePatcher.apply_patch(original, patch)
        assert result.is_valid is True
        applied = result.applied_content

        # Every original section survives verbatim.
        for i in range(1, 23):
            assert f"## Section {i}" in applied
            assert f"detail {i} of the pipeline" in applied
        # The new section was actually added.
        assert "## E2E Test" in applied
        assert "GitHub pull-request workflow" in applied

        # The guard agrees this patch is safe.
        reason = detect_unsafe_additive_rewrite(ADDITIVE_REQUEST, original, applied)
        assert reason is None

        deleted_fraction = compute_deletion_fraction(original, applied)
        assert deleted_fraction < 0.05  # essentially nothing removed


class TestDeterministicGuardRejectsDestructivePatch:
    """2. Additive request with large deletion -> deterministic guard
    rejects it before approval, regardless of LLM verdict."""

    def test_detect_unsafe_additive_rewrite_flags_large_deletion(self):
        original = _readme_134_lines()
        # Mirrors the real bug: ~ -120/+25 - most content dropped, a short
        # new document written in its place.
        destructive_update = (
            "# Agentic AI Software Engineer\n\n"
            "## E2E Test\n\n"
            "This section verifies the autonomous software-engineering "
            "pipeline and GitHub pull-request workflow.\n"
        )
        reason = detect_unsafe_additive_rewrite(ADDITIVE_REQUEST, original, destructive_update)
        assert reason is not None
        assert "%" in reason

    def test_quality_pipeline_check_patch_scope_fails_closed(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        destructive_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="",  # whole-file replacement, like the real bug
            updated_code_snippet=(
                "# Agentic AI Software Engineer\n\n## E2E Test\n\n"
                "This section verifies the autonomous software-engineering "
                "pipeline and GitHub pull-request workflow.\n"
            ),
            explanation="Add E2E Test section",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [destructive_patch], user_request=ADDITIVE_REQUEST
        )
        assert check.status == QualityCheckStatus.FAIL.value
        assert check.category == FailureCategory.PATCH_APPLICATION_FAILURE.value
        assert "README.md" in check.stderr_summary

    def test_judge_rejects_before_approval_even_with_llm_pass(self):
        """The exact real-world failure: LLM QA said PASS. The deterministic
        guard must still fail the candidate, using an existing failure
        category (PATCH_APPLICATION_FAILURE), independent of that verdict."""
        from backend.schemas.qa import QAResult, QualityCheck

        checks = [
            QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
            QualityCheck(
                name="patch_scope",
                status=QualityCheckStatus.FAIL.value,
                stderr_summary="README.md: Request appears additive but the patch deletes 90% ...",
                category=FailureCategory.PATCH_APPLICATION_FAILURE.value,
            ),
            QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
        ]
        optimistic_llm_pass = QAResult(status="PASS", confidence=0.95, summary="Looks complete.")

        qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=optimistic_llm_pass)

        assert qa_result.status == "FAIL"
        assert qa_result.failure_category == FailureCategory.PATCH_APPLICATION_FAILURE.value

        from backend.graph.nodes import qa_router

        state = {"user_message": ADDITIVE_REQUEST, "qa_result": qa_result, "revision_count": 0}
        assert qa_router(state) != "pass"


class TestMixedAdditiveAndTargetedRemovalIsNotABlanketBypass:
    """
    Regression for a classifier gap found in review: a request that mixes
    an addition with a *targeted, singular* removal ("...and remove the
    obsolete section") must NOT be treated as a blanket rewrite
    authorization just because it contains "remove". Only a sweeping,
    plural removal ("remove specified sections") - the literal case the
    requirements listed as an allowed rewrite - should bypass the guard.
    """

    MIXED_REQUEST = "Add an E2E Test section and remove the obsolete section."

    def test_singular_targeted_removal_is_not_an_explicit_rewrite(self):
        assert is_explicit_rewrite_request(self.MIXED_REQUEST) is False
        assert is_additive_request(self.MIXED_REQUEST) is True

    def test_guard_still_rejects_large_deletion_despite_the_word_remove(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        # Only the "obsolete section" was asked to be removed, but this
        # patch (like the real bug) replaces almost the entire file with a
        # short new document - deleting far more than one section's worth.
        destructive_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="",
            updated_code_snippet=(
                "# Agentic AI Software Engineer\n\n## E2E Test\n\n"
                "This section verifies the autonomous software-engineering "
                "pipeline and GitHub pull-request workflow.\n"
            ),
            explanation="Add E2E Test section and remove obsolete section",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [destructive_patch], user_request=self.MIXED_REQUEST
        )
        assert check.status == QualityCheckStatus.FAIL.value
        assert check.category == FailureCategory.PATCH_APPLICATION_FAILURE.value

    def test_plural_sweeping_removal_still_bypasses_as_intended(self):
        """Contrast case: the literal requirement wording ("remove
        specified sections", plural) is still honored as an explicit,
        bounded-by-nothing rewrite signal."""
        assert is_explicit_rewrite_request("Remove the specified sections.") is True

    def test_small_actual_removal_matching_the_request_scope_passes(self, tmp_path):
        """A patch that genuinely only removes the one obsolete section
        (and adds the new one) - proportionate to what was asked - must
        still be allowed."""
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        # Replace exactly one section's worth of content with the new
        # section - a small, proportionate change.
        obsolete_section = "\n".join(original.splitlines()[2:6])
        proportionate_patch = FilePatch(
            file_path="README.md",
            original_code_snippet=obsolete_section,
            updated_code_snippet="## E2E Test\n\nDescribes the pipeline.\n",
            explanation="Remove obsolete section, add E2E Test section",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [proportionate_patch], user_request=self.MIXED_REQUEST
        )
        assert check.status == QualityCheckStatus.PASS.value


class TestExplicitRewriteAllowsLargeDeletion:
    """3. Explicit rewrite request -> large deletion is allowed.
    Also covers 6: the original_code_snippet="" whole-file convention is
    unaffected by this guard."""

    def test_rewrite_request_is_not_flagged(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        rewrite_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="",  # whole-file convention, unchanged semantics
            updated_code_snippet="# Agentic AI Software Engineer\n\nCompletely new content.\n",
            explanation="Full rewrite as requested",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path),
            [rewrite_patch],
            user_request="Please rewrite the entire README from scratch with a simpler structure.",
        )
        assert check.status == QualityCheckStatus.PASS.value

    def test_is_explicit_rewrite_request_detection(self):
        assert is_explicit_rewrite_request("Please rewrite the README.") is True
        assert is_explicit_rewrite_request("Replace the contents of the file.") is True
        assert is_explicit_rewrite_request("Completely restructure this document.") is True
        assert is_explicit_rewrite_request("Regenerate the document.") is True
        assert is_explicit_rewrite_request("Remove the specified sections.") is True
        assert is_explicit_rewrite_request('Add an "E2E Test" section.') is False


class TestNormalSmallEditIsAllowed:
    """4. Normal small README edit -> allowed."""

    def test_small_edit_passes(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        small_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="This is paragraph content for section 1.",
            updated_code_snippet="This is UPDATED paragraph content for section 1.",
            explanation="Fix wording",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [small_patch], user_request="Fix a typo in section 1 of the README."
        )
        assert check.status == QualityCheckStatus.PASS.value

    def test_additive_small_insert_passes(self, tmp_path):
        original = _readme_134_lines()
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        anchor = "\n".join(original.splitlines()[-3:])
        small_additive_patch = FilePatch(
            file_path="README.md",
            original_code_snippet=anchor,
            updated_code_snippet=anchor + "\n## E2E Test\n\nDescribes the pipeline.\n",
            explanation="Add E2E Test section",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [small_additive_patch], user_request=ADDITIVE_REQUEST
        )
        assert check.status == QualityCheckStatus.PASS.value


class TestSafePatcherExactMatchUnchanged:
    """5. Existing SafePatcher exact-match behavior remains unchanged by
    this guard (patcher.py itself was not modified)."""

    def test_mismatched_snippet_still_rejected_by_safepatcher_directly(self):
        patch = FilePatch(
            file_path="README.md",
            original_code_snippet="## Section That Does Not Exist",
            updated_code_snippet="## New Section",
            explanation="Add section",
        )
        result = SafePatcher.apply_patch("# Project\n\nExisting.\n", patch)
        assert result.is_valid is False
        assert "not found" in result.syntax_errors[0]

    def test_check_patch_scope_never_masks_a_safepatcher_mismatch(self, tmp_path):
        """check_patch_scope must not attempt to evaluate (or silently
        pass) a patch whose anchor doesn't even exist in the file -
        check_ast is still the sole authority for that failure."""
        (tmp_path / "README.md").write_text(_readme_134_lines(), encoding="utf-8")
        bad_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="## Section That Does Not Exist Anywhere",
            updated_code_snippet="## E2E Test\n",
            explanation="Add E2E Test section",
        )
        scope_check = QualityPipeline.check_patch_scope(
            str(tmp_path), [bad_patch], user_request=ADDITIVE_REQUEST
        )
        assert scope_check.status == QualityCheckStatus.PASS.value  # not this check's concern

        ast_check = QualityPipeline.check_ast(str(tmp_path), [bad_patch])
        assert ast_check.status == QualityCheckStatus.FAIL.value


class TestAdditiveAndRewriteClassification:
    def test_is_additive_request_matches_expected_phrases(self):
        assert is_additive_request("Add a new section") is True
        assert is_additive_request("Append a changelog entry") is True
        assert is_additive_request("Insert a note at the top") is True
        assert is_additive_request("Extend the FAQ with one more item") is True
        assert is_additive_request("Rewrite the whole document") is False

    def test_compute_deletion_fraction(self):
        original = "a\nb\nc\nd\n"
        # Deletes 3 of 4 original lines.
        updated = "a\nnew\n"
        assert compute_deletion_fraction(original, updated) == 0.75

    def test_compute_deletion_fraction_new_file_is_zero(self):
        assert compute_deletion_fraction("", "brand new content\n") == 0.0


# ============================================================================
# Negated-rewrite-instruction bug: is_explicit_rewrite_request() matched bare
# keywords ("rewrite", "replace", "restructure", "regenerate") even when
# they appeared in a negated instruction like "Do not rewrite the existing
# README content." - causing detect_unsafe_additive_rewrite() to (wrongly)
# treat the request as an explicit rewrite authorization and skip the
# deletion-fraction guard entirely, exactly when a destructive patch is
# handed a request that explicitly tells it not to be destructive.
# ============================================================================

PROD_ADDITIVE_NEGATED_REWRITE_REQUEST = (
    "Add a small E2E Test section to README.md with a short description. "
    "Do not rewrite or replace the existing README content. "
    "Make the smallest additive change possible."
)


class TestNegatedRewriteInstructionIsNotAnExplicitRewrite:
    def test_genuine_rewrite_requests_still_classified_true(self):
        assert is_explicit_rewrite_request("Please rewrite the README.") is True
        assert is_explicit_rewrite_request("Replace the contents of the file.") is True
        assert is_explicit_rewrite_request("Completely restructure this document.") is True
        assert is_explicit_rewrite_request("Regenerate the document.") is True
        assert is_explicit_rewrite_request("Remove the specified sections.") is True

    def test_plain_additive_request_is_false(self):
        assert is_explicit_rewrite_request("Add an E2E Test section.") is False

    def test_negated_rewrite_instructions_are_false(self):
        assert is_explicit_rewrite_request("Do not rewrite the existing README content.") is False
        assert is_explicit_rewrite_request("Do not replace the existing README content.") is False
        assert is_explicit_rewrite_request(
            "Do not rewrite, replace, restructure, or regenerate existing content."
        ) is False

    def test_exact_production_request_is_not_an_explicit_rewrite(self):
        assert is_explicit_rewrite_request(PROD_ADDITIVE_NEGATED_REWRITE_REQUEST) is False

    def test_negation_in_an_earlier_unrelated_sentence_does_not_suppress_a_later_genuine_rewrite(self):
        """A negation cue must only suppress a rewrite keyword in the SAME
        clause - an unrelated earlier negation must never mask a genuine,
        later rewrite request."""
        text = "Do not worry about formatting. Please rewrite the README from scratch."
        assert is_explicit_rewrite_request(text) is True

    def test_detect_unsafe_additive_rewrite_still_flags_destructive_patch_under_negated_request(self):
        """The exact production scenario: an additive request that
        explicitly forbids rewriting/replacing must still trigger the
        deletion-fraction guard against a destructive whole-file
        replacement - the guard must not be bypassed just because the
        word "rewrite" appears (negated) in the request."""
        original = "# agentic-ai-test-repo"
        destructive_update = (
            "# Project Title\n\n"
            "*(Existing project description goes here.)*\n\n"
            "## E2E Test\nShort description.\n"
        )
        reason = detect_unsafe_additive_rewrite(
            PROD_ADDITIVE_NEGATED_REWRITE_REQUEST, original, destructive_update
        )
        assert reason is not None
        assert "%" in reason

    def test_detect_unsafe_additive_rewrite_still_allows_genuine_minimal_additive_patch(self):
        """Sanity check in the other direction: a genuinely minimal,
        content-preserving patch under the same negated-rewrite request
        must still be allowed."""
        original = "# agentic-ai-test-repo"
        safe_update = "# agentic-ai-test-repo\n\n## E2E Test\nShort description.\n"
        reason = detect_unsafe_additive_rewrite(
            PROD_ADDITIVE_NEGATED_REWRITE_REQUEST, original, safe_update
        )
        assert reason is None

    def test_quality_pipeline_check_patch_scope_fails_closed_under_negated_request(self, tmp_path):
        """Full check_patch_scope integration: the exact production
        request + a substantial destructive replacement of the original
        README must be a FAIL (PATCH_APPLICATION_FAILURE), never a PASS."""
        original = "# agentic-ai-test-repo"
        (tmp_path / "README.md").write_text(original, encoding="utf-8")

        destructive_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="",
            updated_code_snippet=(
                "# Project Title\n\n"
                "*(Existing project description goes here.)*\n\n"
                "## E2E Test\nShort description.\n"
            ),
            explanation="Add E2E Test section",
        )

        check = QualityPipeline.check_patch_scope(
            str(tmp_path), [destructive_patch], user_request=PROD_ADDITIVE_NEGATED_REWRITE_REQUEST,
        )
        assert check.status == QualityCheckStatus.FAIL.value
        assert check.category == FailureCategory.PATCH_APPLICATION_FAILURE.value


class TestDeveloperPromptGuidance:
    """Requirement B: the developer prompt explicitly instructs minimal,
    content-preserving patches for additive requests."""

    def test_developer_node_prompt_instructs_minimal_additive_patches(self, tmp_path, monkeypatch):
        import os
        import subprocess

        from backend.graph.nodes import developer_node
        from backend.indexer.models import CodeChunk
        from backend.schemas.developer import DeveloperResult

        project_id = "patch-scope-prompt-test"
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)
        (workspace_dir / "README.md").write_text(_readme_134_lines(), encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        captured = {}

        class _FakePatchResult:
            patches = [
                FilePatch(
                    file_path="README.md",
                    original_code_snippet="",
                    updated_code_snippet="# New\n",
                    explanation="stub",
                )
            ]

        def capturing_invoke_structured(llm, schema_cls, prompt, *a, **k):
            captured["prompt"] = prompt
            return _FakePatchResult()

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="stub", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr("backend.graph.nodes.invoke_structured", capturing_invoke_structured)

        state = {
            "user_message": ADDITIVE_REQUEST,
            "project_id": project_id,
            "repo_context": [
                CodeChunk(
                    file_path="README.md",
                    content=_readme_134_lines(),
                    start_line=1,
                    end_line=134,
                    chunk_type="module",
                )
            ],
        }

        developer_node(state)

        prompt = captured["prompt"]
        assert "preserve all unrelated existing content" in prompt.lower()
        assert "do not remove existing sections" in prompt.lower()
        assert "explicitly asks to rewrite" in prompt.lower() or "explicitly ask for a rewrite" in prompt.lower()


# ---------------------------------------------------------------------------
# Real E2E regression: run_558b917f75d1
#
# patch_scope reported PASS on a real, live run with the exact bug-report
# request against a real 134-line README and a real destructive patch
# (+25/-120, 83% deletion). Root cause: developer_node writes the patched
# content to repo_path BEFORE qa_node's checks ever run. check_patch_scope
# then re-read "original" straight off that same, already-mutated disk
# path - comparing the destructive content against itself and finding 0%
# deletion. Every prior unit test (including this file's own) built a
# pristine tmp_path and called check_patch_scope directly, without ever
# reproducing that write-then-check ordering - which is exactly why they
# all passed while the real run didn't.
#
# Fixed by having developer_node/revision_node record each file's true
# pre-write content (AgentState["pre_patch_snapshots"]) and having
# check_patch_scope/run_all prefer that snapshot over a live disk read.
#
# These tests exercise the REAL developer_node (real git workspace, real
# disk write) followed by the REAL QualityPipeline/StructuredQAJudge, not
# hand-constructed fixtures, so they fail exactly the way the real run did
# if the write-then-check ordering bug ever comes back.
# ---------------------------------------------------------------------------

def _real_git_workspace_with_readme(tmp_path, monkeypatch, project_id: str, readme_content: str):
    import os
    import subprocess

    workspace_dir = tmp_path / "workspace" / "default-org" / project_id
    workspace_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)
    (workspace_dir / "README.md").write_text(readme_content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    return workspace_dir


def _run_real_developer_node(tmp_path, monkeypatch, project_id, user_message, readme_content, destructive_patch):
    from backend.graph.nodes import developer_node
    from backend.indexer.models import CodeChunk
    from backend.schemas.developer import DeveloperResult

    _real_git_workspace_with_readme(tmp_path, monkeypatch, project_id, readme_content)

    class _FakePatchResult:
        patches = [destructive_patch]

    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary="stub", changes=[], requires_testing=True, notes=[]
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.invoke_structured",
        lambda llm, schema_cls, prompt, *a, **k: _FakePatchResult(),
    )

    state = {
        "user_message": user_message,
        "project_id": project_id,
        "repo_context": [
            CodeChunk(
                file_path="README.md",
                content=readme_content,
                start_line=1,
                end_line=len(readme_content.splitlines()),
                chunk_type="module",
            )
        ],
    }
    return developer_node(state)


_DESTRUCTIVE_README_PATCH = FilePatch(
    file_path="README.md",
    original_code_snippet="",  # whole-file convention - matches the real run
    updated_code_snippet=(
        "# Agentic AI Software Engineer\n\n## E2E Test\n\n"
        "This section verifies the autonomous software-engineering "
        "pipeline and GitHub pull-request workflow.\n"
    ),
    explanation="Add E2E Test section",
)


class TestRealE2EWriteThenCheckOrderingBug:
    """A. Exact 134-line README + exact additive request + a destructive
    (~98% deletion) whole-file patch, driven through the REAL developer_node
    (which really writes the patched content to disk) -> patch_scope must
    still FAIL, because it must consult the pre-write snapshot, not disk."""

    def test_real_developer_node_write_then_check_patch_scope_fails(self, tmp_path, monkeypatch):
        original = _readme_134_lines()
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-repro", ADDITIVE_REQUEST, original, _DESTRUCTIVE_README_PATCH
        )

        # developer_node really did write the destructive content to disk,
        # exactly like production - this is the condition that broke the
        # old (disk-read-only) implementation.
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-repro" / "README.md"
        assert readme_path.read_text(encoding="utf-8") == _DESTRUCTIVE_README_PATCH.updated_code_snippet

        pre_patch_snapshots = output["pre_patch_snapshots"]
        assert pre_patch_snapshots["README.md"] == original  # true original, not the mutated disk content

        check = QualityPipeline.check_patch_scope(
            str(readme_path.parent),
            output["generated_patches"],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=pre_patch_snapshots,
        )
        assert check.status == QualityCheckStatus.FAIL.value
        assert check.category == FailureCategory.PATCH_APPLICATION_FAILURE.value

    def test_without_snapshot_reproduces_the_original_bug(self, tmp_path, monkeypatch):
        """Proves the fix is actually load-bearing: omitting the snapshot
        (the old call signature/behavior) reproduces the exact false PASS
        from run_558b917f75d1."""
        original = _readme_134_lines()
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-repro-no-snapshot", ADDITIVE_REQUEST, original, _DESTRUCTIVE_README_PATCH
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-repro-no-snapshot" / "README.md"

        check_without_snapshot = QualityPipeline.check_patch_scope(
            str(readme_path.parent), output["generated_patches"], user_request=ADDITIVE_REQUEST
        )
        assert check_without_snapshot.status == QualityCheckStatus.PASS.value  # the bug, reproduced


class TestRealE2EThroughRunAll:
    """B. Same case through QualityPipeline.run_all() -> FAIL."""

    def test_run_all_fails_on_the_real_destructive_patch(self, tmp_path, monkeypatch):
        original = _readme_134_lines()
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-run-all", ADDITIVE_REQUEST, original, _DESTRUCTIVE_README_PATCH
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-run-all" / "README.md"

        checks, _ = QualityPipeline.run_all(
            str(readme_path.parent),
            output["generated_patches"],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=output["pre_patch_snapshots"],
        )
        by_name = {c.name: c for c in checks}
        assert by_name["patch_scope"].status == QualityCheckStatus.FAIL.value
        assert by_name["patch_scope"].category == FailureCategory.PATCH_APPLICATION_FAILURE.value


class TestRealE2EThroughJudge:
    """C. Same case through judge/B2 -> FAIL even if the LLM says PASS."""

    def test_judge_fails_despite_optimistic_llm_review(self, tmp_path, monkeypatch):
        from backend.schemas.qa import QAResult

        original = _readme_134_lines()
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-judge", ADDITIVE_REQUEST, original, _DESTRUCTIVE_README_PATCH
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-judge" / "README.md"

        checks, test_result = QualityPipeline.run_all(
            str(readme_path.parent),
            output["generated_patches"],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=output["pre_patch_snapshots"],
        )
        optimistic_llm_pass = QAResult(status="PASS", confidence=0.95, summary="Looks complete and correct.")

        qa_result = StructuredQAJudge.evaluate(checks, test_result=test_result, llm_qa_result=optimistic_llm_pass)

        assert qa_result.status == "FAIL"
        assert qa_result.failure_category == FailureCategory.PATCH_APPLICATION_FAILURE.value


class TestRealE2EExplicitRewriteAllowed:
    """D. Explicit rewrite request -> allowed, through the real
    developer_node write-then-check sequence."""

    def test_explicit_rewrite_passes_through_real_sequence(self, tmp_path, monkeypatch):
        original = _readme_134_lines()
        rewrite_request = "Please rewrite the entire README from scratch with a simpler structure."
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-rewrite", rewrite_request, original, _DESTRUCTIVE_README_PATCH
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-rewrite" / "README.md"

        checks, _ = QualityPipeline.run_all(
            str(readme_path.parent),
            output["generated_patches"],
            user_request=rewrite_request,
            original_file_snapshots=output["pre_patch_snapshots"],
        )
        by_name = {c.name: c for c in checks}
        assert by_name["patch_scope"].status == QualityCheckStatus.PASS.value


class TestRealE2EMixedAdditiveAndTargetedRemoval:
    """E. Mixed additive + single targeted removal, but the actual patch is
    still a destructive rewrite -> FAIL, through the real sequence."""

    def test_mixed_request_with_destructive_patch_still_fails(self, tmp_path, monkeypatch):
        original = _readme_134_lines()
        mixed_request = "Add an E2E Test section and remove the obsolete section."
        output = _run_real_developer_node(
            tmp_path, monkeypatch, "e2e-bug-mixed", mixed_request, original, _DESTRUCTIVE_README_PATCH
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-bug-mixed" / "README.md"

        checks, _ = QualityPipeline.run_all(
            str(readme_path.parent),
            output["generated_patches"],
            user_request=mixed_request,
            original_file_snapshots=output["pre_patch_snapshots"],
        )
        by_name = {c.name: c for c in checks}
        assert by_name["patch_scope"].status == QualityCheckStatus.FAIL.value


# ---------------------------------------------------------------------------
# Real E2E regression #2: run_d36ddab60993
#
# Even after the pre_patch_snapshots fix (commit 8bb3fde), a second real run
# still got patch_scope=PASS on a ~79% deletion patch. Root cause this time:
# generate_revision_patches() (backend/agents/revision.py) - the
# CONTEXT-AWARE revision path - never writes to disk; it validates its
# candidate's original_code_snippet anchor against CURRENT LIVE disk
# content and stops there. revision_node was carrying the FIRST attempt's
# pre_patch_snapshots forward unconditionally into every later cycle. When
# a later revision's candidate patch has an anchor that exists in the
# current (already-once-modified) disk content but NOT in that stale,
# carried-forward snapshot, check_patch_scope's SafePatcher.apply_patch(
# stale_snapshot, patch) reports the patch invalid (anchor not found) and
# check_patch_scope silently skips it ("check_ast's concern") - while
# check_ast itself validates fine, because it always reads live disk.
# Net effect: patch_scope PASS, AST PASS, on a genuinely destructive patch.
#
# Fixed by having revision_node drop any stale snapshot entry for files a
# context-aware-produced candidate touches, so check_patch_scope falls back
# to the same live-disk read check_ast and generate_revision_patches
# already use - the correct baseline for a patch that was never written
# anywhere.
# ---------------------------------------------------------------------------

class TestRealE2ERevisionCycleStaleSnapshotBug:
    def test_revision_node_with_stale_snapshot_and_live_disk_mismatch_reproduced_directly(self, tmp_path):
        """
        Direct reproduction of the exact runtime discrepancy (without the
        full revision_node plumbing) - proves the mechanism precisely:
        a patch that validates against LIVE disk (as check_ast and
        generate_revision_patches both check) but not against a STALE
        snapshot from an earlier attempt gets silently skipped by
        check_patch_scope, producing a false PASS on a ~97% deletion.
        This is exactly commit 8bb3fde's residual bug - it would have
        FAILED (reported PASS) against that commit.
        """
        from backend.developer.patcher import SafePatcher

        original = _readme_134_lines()
        # Attempt 1 already wrote a slightly different (but still safe)
        # version to disk - a real, legitimate small prior change.
        attempt1_output = (
            "# Agentic AI Software Engineer\n\n(attempt 1 safe note)\n\n"
            + "\n".join(original.splitlines()[1:])
            + "\n"
        )
        (tmp_path / "README.md").write_text(attempt1_output, encoding="utf-8")

        # generate_revision_patches's candidate: anchor exists in CURRENT
        # disk (attempt1_output) - its own validation would pass - but the
        # anchor text ("(attempt 1 safe note)...") never existed in the
        # TRUE original at all.
        anchor = "(attempt 1 safe note)\n\n" + "\n".join(attempt1_output.splitlines()[4:])
        destructive_revision_patch = FilePatch(
            file_path="README.md",
            original_code_snippet=anchor,
            updated_code_snippet="## E2E Test\n\nDescribes the pipeline.\n",
            explanation="revision: add E2E Test section",
        )

        # Sanity: this is exactly what generate_revision_patches's own
        # validation loop and check_ast both do - validate against LIVE
        # disk - and it passes, matching the real run's observed AST PASS.
        live_validation = SafePatcher.apply_patch(attempt1_output, destructive_revision_patch)
        assert live_validation.is_valid is True
        ast_check = QualityPipeline.check_ast(str(tmp_path), [destructive_revision_patch])
        assert ast_check.status == QualityCheckStatus.PASS.value

        stale_snapshot_from_attempt_1 = {"README.md": original}  # true original, now stale

        # THE BUG (commit 8bb3fde behavior): carrying the stale snapshot
        # forward makes check_patch_scope silently skip the patch.
        check_with_stale_snapshot = QualityPipeline.check_patch_scope(
            str(tmp_path),
            [destructive_revision_patch],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=stale_snapshot_from_attempt_1,
        )
        assert check_with_stale_snapshot.status == QualityCheckStatus.PASS.value  # the bug, reproduced

        # THE FIX: dropping the stale entry (what revision_node now does
        # for context-aware-produced candidates) makes check_patch_scope
        # fall back to a live disk read and correctly reject it.
        check_without_stale_entry = QualityPipeline.check_patch_scope(
            str(tmp_path),
            [destructive_revision_patch],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots={},
        )
        assert check_without_stale_entry.status == QualityCheckStatus.FAIL.value
        assert check_without_stale_entry.category == FailureCategory.PATCH_APPLICATION_FAILURE.value

    def test_real_revision_node_drops_stale_snapshot_for_context_aware_candidate(self, tmp_path, monkeypatch):
        """
        Full integration: drives the REAL revision_node() - not a manual
        simulation - through the exact scenario above, and confirms its
        returned pre_patch_snapshots no longer contains the stale entry,
        so the QA cycle that follows evaluates the candidate correctly.
        """
        import os
        import subprocess

        from backend.graph.nodes import revision_node
        from backend.indexer.models import CodeChunk
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan
        from backend.schemas.qa import QAResult

        project_id = "e2e-revision-stale-snapshot"
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)

        original = _readme_134_lines()
        attempt1_output = (
            "# Agentic AI Software Engineer\n\n(attempt 1 safe note)\n\n"
            + "\n".join(original.splitlines()[1:])
            + "\n"
        )
        (workspace_dir / "README.md").write_text(attempt1_output, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(["git", "commit", "-m", "attempt 1"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        anchor = "(attempt 1 safe note)\n\n" + "\n".join(attempt1_output.splitlines()[4:])
        destructive_revision_patch = FilePatch(
            file_path="README.md",
            original_code_snippet=anchor,
            updated_code_snippet="## E2E Test\n\nDescribes the pipeline.\n",
            explanation="revision: add E2E Test section",
        )

        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: DeveloperResult(
                summary="revised", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr(
            "backend.agents.revision.generate_revision_patches",
            lambda **kwargs: [destructive_revision_patch],
        )

        state = {
            "user_message": ADDITIVE_REQUEST,
            "project_id": project_id,
            "plan": ExecutionPlan(goal=ADDITIVE_REQUEST, steps=[], success_criteria="Done"),
            "developer_result": DeveloperResult(summary="attempt 1", changes=[], requires_testing=True, notes=[]),
            "generated_patches": [
                FilePatch(
                    file_path="README.md",
                    original_code_snippet="",
                    updated_code_snippet=attempt1_output,
                    explanation="attempt 1",
                )
            ],
            "qa_result": QAResult(status="FAIL", summary="attempt 1 rejected"),
            "revision_count": 0,
            # Carried over from the FIRST developer_node call - the true,
            # now-stale, pre-attempt-1 original.
            "pre_patch_snapshots": {"README.md": original},
            "repo_context": [
                CodeChunk(
                    file_path="README.md", content=attempt1_output, start_line=1,
                    end_line=len(attempt1_output.splitlines()), chunk_type="module",
                )
            ],
        }

        output = revision_node(state)

        assert output["generated_patches"] == [destructive_revision_patch]
        # The stale entry must be gone - not merely unused, actually absent -
        # so check_patch_scope is forced to fall back to a live disk read.
        assert "README.md" not in output["pre_patch_snapshots"]

        checks, _ = QualityPipeline.run_all(
            str(workspace_dir),
            output["generated_patches"],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=output["pre_patch_snapshots"],
        )
        by_name = {c.name: c for c in checks}
        assert by_name["patch_scope"].status == QualityCheckStatus.FAIL.value
        assert by_name["patch_scope"].category == FailureCategory.PATCH_APPLICATION_FAILURE.value


# ============================================================================
# run_b1c04b69ba54 investigation: a SECOND (or later) whole-file-replacement
# revision candidate slipped through patch_scope as PASS. Root cause: an
# earlier, ALSO-rejected revision attempt had already written its own
# fabricated content to disk (via the blind fallback path). Once that
# happened, popping the snapshot for a later context-aware candidate (the
# fix for TestRealE2ERevisionCycleStaleSnapshotBug above) made
# check_patch_scope fall back to a live disk read - but "live disk" was no
# longer the true original, it was the EARLIER rejected attempt's own
# fabrication. Comparing one fabricated whole-file rewrite against another
# (rather than against the real original) can show a low/safe deletion
# fraction even though both are 100% destructive relative to the truth,
# because SafePatcher.apply_patch always accepts an empty
# original_code_snippet regardless of what "original" is.
#
# Fix: AgentState.true_original_snapshots is captured once (developer_node's
# first read of a file) and never overwritten/popped for the rest of the
# run - check_patch_scope's destructiveness measurement always compares
# against it, independently of whatever baseline validated the candidate.
# ============================================================================

class TestRealE2EMultiCycleFabricationBug:
    def _real_git_repo(self, path, content):
        import subprocess
        path.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(path), capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(path), capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(path), capture_output=True, text=True)
        (path / "README.md").write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, text=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), capture_output=True, text=True, check=True)

    def test_disk_fallback_without_true_original_reproduces_the_bug(self, tmp_path):
        """A. Direct reproduction of the exact mechanism, without the full
        revision_node plumbing: once disk holds an earlier REJECTED
        whole-file-replacement attempt, comparing a second, similarly
        generic fabrication against that (instead of the real original)
        via a popped/absent true-original baseline lets it through as
        PASS - proving the vulnerability existed."""
        repo = tmp_path / "repo"
        self._real_git_repo(repo, "# agentic-ai-test-repo")

        # Attempt 1 (rejected, but already written to disk by the blind
        # fallback path before rejection).
        attempt1 = "# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v1.\n"
        (repo / "README.md").write_text(attempt1, encoding="utf-8")

        # Attempt 2: independently fabricated, but similar generic
        # scaffolding to attempt 1 - 100% destructive relative to the true
        # 1-line original, but low deletion relative to attempt 1.
        attempt2 = FilePatch(
            file_path="README.md", original_code_snippet="",
            updated_code_snippet="# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v2, reworded.\n",
            explanation="revised patch attempt 2",
        )

        # The bug: no true-original baseline recorded, only a (popped/absent)
        # original_file_snapshots entry - falls back to the now-contaminated
        # live disk (attempt 1's content) for the destructiveness comparison.
        buggy_check = QualityPipeline.check_patch_scope(
            str(repo), [attempt2], user_request=ADDITIVE_REQUEST, original_file_snapshots={},
        )
        assert buggy_check.status == QualityCheckStatus.PASS.value  # the bug, reproduced

    def test_true_original_snapshot_fixes_the_bug(self, tmp_path):
        """B. The fix: passing true_original_snapshots (the real,
        run-lifetime original, captured once) makes check_patch_scope
        correctly reject attempt 2 as destructive, regardless of what
        disk currently holds."""
        repo = tmp_path / "repo"
        self._real_git_repo(repo, "# agentic-ai-test-repo")

        attempt1 = "# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v1.\n"
        (repo / "README.md").write_text(attempt1, encoding="utf-8")

        attempt2 = FilePatch(
            file_path="README.md", original_code_snippet="",
            updated_code_snippet="# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v2, reworded.\n",
            explanation="revised patch attempt 2",
        )

        fixed_check = QualityPipeline.check_patch_scope(
            str(repo), [attempt2], user_request=ADDITIVE_REQUEST,
            original_file_snapshots={},
            true_original_snapshots={"README.md": "# agentic-ai-test-repo"},
        )
        assert fixed_check.status == QualityCheckStatus.FAIL.value
        assert fixed_check.category == FailureCategory.PATCH_APPLICATION_FAILURE.value

    def test_real_multi_cycle_revision_node_preserves_true_original_and_catches_fabrication(self, tmp_path, monkeypatch):
        """C. Full integration: drives the REAL developer_node-failure ->
        revision cycle 1 (blind fallback, writes to disk, rejected) ->
        revision cycle 2 (context-aware, independently fabricated,
        similar-looking whole-file replacement) - and confirms the final
        QA evaluation still correctly rejects it, because
        true_original_snapshots survived both cycles unmodified."""
        import os
        from backend.graph.nodes import revision_node
        from backend.schemas.developer import DeveloperResult, FileChange
        from backend.schemas.qa import QAResult

        project_id = "e2e-multi-cycle-fabrication"
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        self._real_git_repo(workspace_dir, "# agentic-ai-test-repo")
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        base_state = {
            "user_message": ADDITIVE_REQUEST,
            "project_id": project_id,
            "qa_result": QAResult(status="FAIL", summary="anchor mismatch"),
            "generated_patches": [],
            "developer_result": DeveloperResult(summary="stub", changes=[], requires_testing=True, notes=[]),
            "repo_context": None,
            "revision_count": 0,
            "pre_patch_snapshots": {"README.md": "# agentic-ai-test-repo"},
            "true_original_snapshots": {"README.md": "# agentic-ai-test-repo"},
        }

        attempt1_content = "# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v1.\n"
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda **k: DeveloperResult(
                summary="revised", changes=[FileChange(file_path="README.md", action="MODIFY", content=attempt1_content, reasoning="x")],
                requires_testing=True, notes=[],
            ),
        )
        cycle1_output = revision_node(base_state)

        assert (workspace_dir / "README.md").read_text(encoding="utf-8") == attempt1_content
        assert cycle1_output["true_original_snapshots"]["README.md"] == "# agentic-ai-test-repo"

        attempt2_patch = FilePatch(
            file_path="README.md", original_code_snippet="",
            updated_code_snippet="# Project Title\n\n*(Existing project description goes here.)*\n\n## E2E Test\nSection body v2, reworded.\n",
            explanation="revised patch attempt 2",
        )
        cycle2_state = {
            **base_state,
            "qa_result": QAResult(status="FAIL", summary="attempt 1 rejected"),
            "generated_patches": cycle1_output["generated_patches"],
            "developer_result": cycle1_output["developer_result"],
            "repo_context": [1],
            "revision_count": 1,
            "pre_patch_snapshots": cycle1_output["pre_patch_snapshots"],
            "true_original_snapshots": cycle1_output["true_original_snapshots"],
        }
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda **k: DeveloperResult(summary="revised2", changes=[], requires_testing=True, notes=[]),
        )
        monkeypatch.setattr(
            "backend.agents.revision.generate_revision_patches",
            lambda **k: [attempt2_patch],
        )
        cycle2_output = revision_node(cycle2_state)

        # True original must have survived both cycles unchanged.
        assert cycle2_output["true_original_snapshots"]["README.md"] == "# agentic-ai-test-repo"

        checks, _ = QualityPipeline.run_all(
            str(workspace_dir),
            cycle2_output["generated_patches"],
            user_request=ADDITIVE_REQUEST,
            original_file_snapshots=cycle2_output["pre_patch_snapshots"],
            true_original_snapshots=cycle2_output["true_original_snapshots"],
        )
        by_name = {c.name: c for c in checks}
        assert by_name["patch_scope"].status == QualityCheckStatus.FAIL.value
        assert by_name["patch_scope"].category == FailureCategory.PATCH_APPLICATION_FAILURE.value
        assert by_name["patch_scope"].category == FailureCategory.PATCH_APPLICATION_FAILURE.value

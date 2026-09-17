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

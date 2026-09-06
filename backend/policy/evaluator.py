from datetime import datetime, timezone
import fnmatch
from typing import Any, Dict, List, Optional

from backend.policy.path_filter import (
    is_traversal_attack,
    matches_protected_path,
    normalize_path,
)
from backend.policy.patterns import (
    DEPENDENCY_MANIFEST_PATTERNS as DEPENDENCY_PATTERNS,
    CI_CD_PATTERNS as CI_PATTERNS,
    INFRA_CONFIG_PATTERNS as CONFIG_PATTERNS,
)
from backend.schemas.policy import (
    PolicyConfig,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyViolation,
    PolicyTelemetry,
)
from backend.schemas.qa import QAResult


class PolicyEvaluator:
    """
    Deterministic organization policy evaluation engine.
    Evaluates repositories, branches, protected paths, patch limits,
    required quality checks, and risk thresholds before external mutation.
    """

    @classmethod
    def evaluate(
        cls,
        policy: Optional[PolicyConfig] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        changed_files: Optional[List[str]] = None,
        lines_added: int = 0,
        lines_deleted: int = 0,
        risk_score: str = "LOW",
        risk_reasons: Optional[List[str]] = None,
        quality_results: Optional[QAResult] = None,
        revision_count: int = 0,
        model: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        if policy is None:
            policy = PolicyConfig()

        changed_files = changed_files or []
        risk_reasons = risk_reasons or []
        norm_risk = (risk_score or "LOW").strip().upper()

        violations: List[PolicyViolation] = []
        warnings: List[str] = []
        checks: Dict[str, str] = {}

        # -----------------------------------------------------------------
        # 1. Repository Boundary Check
        # -----------------------------------------------------------------
        if policy.allowed_repositories is not None:
            if not repository:
                violations.append(
                    PolicyViolation(
                        rule="REPOSITORY_UNDEFINED",
                        message="Repository is required by organization policy but was not specified.",
                    )
                )
                checks["repository"] = "FAIL"
            else:
                is_repo_allowed = any(
                    fnmatch.fnmatch(repository.lower(), allowed.lower())
                    for allowed in policy.allowed_repositories
                )
                if not is_repo_allowed:
                    violations.append(
                        PolicyViolation(
                            rule="ALLOWED_REPOSITORIES",
                            message=f"Repository '{repository}' is not in the allowed repositories list.",
                            target=repository,
                        )
                    )
                    checks["repository"] = "FAIL"
                else:
                    checks["repository"] = "PASS"
        else:
            checks["repository"] = "PASS"

        # -----------------------------------------------------------------
        # 2. Branch Boundary & Protected Branches
        # -----------------------------------------------------------------
        branch_failed = False
        if branch:
            norm_branch = branch.strip().lower()
            # Check protected branches
            for pb in policy.protected_branches:
                if fnmatch.fnmatch(norm_branch, pb.lower()):
                    violations.append(
                        PolicyViolation(
                            rule="PROTECTED_BRANCH",
                            message=f"Target branch '{branch}' is a protected branch and cannot be directly targeted.",
                            target=branch,
                        )
                    )
                    branch_failed = True
                    break

            # Check allowed branches
            if policy.allowed_branches is not None and not branch_failed:
                is_branch_allowed = any(
                    fnmatch.fnmatch(norm_branch, ab.lower())
                    for ab in policy.allowed_branches
                )
                if not is_branch_allowed:
                    violations.append(
                        PolicyViolation(
                            rule="ALLOWED_BRANCHES",
                            message=f"Branch '{branch}' is not in the allowed branches list.",
                            target=branch,
                        )
                    )
                    branch_failed = True

        checks["branch"] = "FAIL" if branch_failed else "PASS"

        # -----------------------------------------------------------------
        # 3. Path Traversal, Protected Paths, & File Change Permissions
        # -----------------------------------------------------------------
        path_failed = False
        for f in changed_files:
            # Traversal check
            if is_traversal_attack(f):
                violations.append(
                    PolicyViolation(
                        rule="PATH_TRAVERSAL",
                        message=f"Path traversal detected in file path: {f}",
                        target=f,
                    )
                )
                path_failed = True
                continue

            # Protected paths check
            is_protected, pattern = matches_protected_path(f, policy.protected_paths)
            if is_protected:
                violations.append(
                    PolicyViolation(
                        rule="PROTECTED_PATH",
                        message=f"File '{f}' matches protected path pattern '{pattern}'.",
                        target=f,
                    )
                )
                path_failed = True

            norm_f = normalize_path(f).lower()

            # Dependency changes permission
            if not policy.allow_dependency_changes:
                if any(norm_f.endswith(dp.lower()) or dp.lower() in norm_f for dp in DEPENDENCY_PATTERNS):
                    violations.append(
                        PolicyViolation(
                            rule="DEPENDENCY_CHANGES_DISALLOWED",
                            message=f"Dependency manifest modification is not permitted by policy: {f}",
                            target=f,
                        )
                    )
                    path_failed = True

            # CI/CD changes permission
            if not policy.allow_ci_changes:
                if any(cip.lower() in norm_f for cip in CI_PATTERNS):
                    violations.append(
                        PolicyViolation(
                            rule="CI_CHANGES_DISALLOWED",
                            message=f"CI/CD pipeline modification is not permitted by policy: {f}",
                            target=f,
                        )
                    )
                    path_failed = True

            # Config changes permission
            if not policy.allow_config_changes:
                if any(cp.lower() in norm_f for cp in CONFIG_PATTERNS):
                    violations.append(
                        PolicyViolation(
                            rule="CONFIG_CHANGES_DISALLOWED",
                            message=f"Infrastructure configuration modification is not permitted by policy: {f}",
                            target=f,
                        )
                    )
                    path_failed = True

        checks["protected_paths"] = "FAIL" if path_failed else "PASS"

        # -----------------------------------------------------------------
        # 4. Patch Limits (Files, Lines Added, Lines Deleted, Revisions)
        # -----------------------------------------------------------------
        limits_failed = False
        if len(changed_files) > policy.max_files_changed:
            violations.append(
                PolicyViolation(
                    rule="MAX_FILES_EXCEEDED",
                    message=(
                        f"Files changed ({len(changed_files)}) exceeds "
                        f"policy maximum ({policy.max_files_changed})."
                    ),
                    target=str(len(changed_files)),
                )
            )
            limits_failed = True

        if lines_added > policy.max_lines_added:
            violations.append(
                PolicyViolation(
                    rule="MAX_LINES_ADDED_EXCEEDED",
                    message=(
                        f"Lines added ({lines_added}) exceeds "
                        f"policy maximum ({policy.max_lines_added})."
                    ),
                    target=str(lines_added),
                )
            )
            limits_failed = True

        if lines_deleted > policy.max_lines_deleted:
            violations.append(
                PolicyViolation(
                    rule="MAX_LINES_DELETED_EXCEEDED",
                    message=(
                        f"Lines deleted ({lines_deleted}) exceeds "
                        f"policy maximum ({policy.max_lines_deleted})."
                    ),
                    target=str(lines_deleted),
                )
            )
            limits_failed = True

        if revision_count > policy.max_revisions:
            violations.append(
                PolicyViolation(
                    rule="MAX_REVISIONS_EXCEEDED",
                    message=(
                        f"Revision cycles ({revision_count}) exceeds "
                        f"policy maximum ({policy.max_revisions})."
                    ),
                    target=str(revision_count),
                )
            )
            limits_failed = True

        checks["patch_limits"] = "FAIL" if limits_failed else "PASS"

        # -----------------------------------------------------------------
        # 5. Required Quality Checks
        # -----------------------------------------------------------------
        quality_failed = False
        if quality_results is not None:
            if quality_results.status == "FAIL":
                violations.append(
                    PolicyViolation(
                        rule="REQUIRED_CHECK_FAILED",
                        message=f"QA pipeline failed: {quality_results.summary}",
                        target=quality_results.failure_category or "QA_FAILURE",
                    )
                )
                quality_failed = True
            elif getattr(quality_results, "checks", None):
                check_map = {c.name.upper(): c.status for c in quality_results.checks}
                for req in policy.required_quality_checks:
                    req_upper = req.upper()
                    st = check_map.get(req_upper)
                    if st == "FAIL":
                        violations.append(
                            PolicyViolation(
                                rule="REQUIRED_CHECK_FAILED",
                                message=f"Required check '{req}' failed in QA pipeline.",
                                target=req,
                            )
                        )
                        quality_failed = True

        checks["required_checks"] = "FAIL" if quality_failed else "PASS"

        # -----------------------------------------------------------------
        # 6. Model Allowlist
        # -----------------------------------------------------------------
        if policy.allowed_models and model:
            if not any(fnmatch.fnmatch(model.lower(), m.lower()) for m in policy.allowed_models):
                violations.append(
                    PolicyViolation(
                        rule="ALLOWED_MODELS",
                        message=f"Model '{model}' is not in the allowed models list.",
                        target=model,
                    )
                )

        # -----------------------------------------------------------------
        # 7. Risk Threshold Integration
        # -----------------------------------------------------------------
        risk_decision = policy.risk_thresholds.get(norm_risk, "REVIEW").upper()
        if norm_risk == "CRITICAL" or risk_decision == "BLOCK":
            violations.append(
                PolicyViolation(
                    rule="CRITICAL_RISK_BLOCKED",
                    message=f"Changeset assessed as CRITICAL risk: {'; '.join(risk_reasons)}",
                    target=norm_risk,
                )
            )
            checks["risk_policy"] = "BLOCK"
        elif risk_decision == "REVIEW" or norm_risk in ("MEDIUM", "HIGH"):
            warnings.append(
                f"Assessed risk level '{norm_risk}' requires human review: {'; '.join(risk_reasons)}"
            )
            checks["risk_policy"] = "REVIEW"
        else:
            checks["risk_policy"] = "PASS"

        # -----------------------------------------------------------------
        # Final Decision Synthesis
        # -----------------------------------------------------------------
        if violations:
            final_decision = PolicyDecision.BLOCK
        elif warnings or checks.get("risk_policy") == "REVIEW":
            final_decision = PolicyDecision.REVIEW
        else:
            final_decision = PolicyDecision.ALLOW

        now_iso = datetime.now(timezone.utc).isoformat()

        return PolicyEvaluationResult(
            decision=final_decision,
            violations=violations,
            warnings=warnings,
            checks=checks,
            policy_version=policy.policy_version,
            evaluated_at=now_iso,
        )

    @classmethod
    def create_telemetry(
        cls,
        result: PolicyEvaluationResult,
        risk_score: str,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        changed_files: Optional[List[str]] = None,
        lines_added: int = 0,
        lines_deleted: int = 0,
        required_checks: Optional[List[str]] = None,
        failed_checks: Optional[List[str]] = None,
    ) -> PolicyTelemetry:
        return PolicyTelemetry(
            policy_version=result.policy_version,
            decision=result.decision.value,
            violations=[v.message for v in result.violations],
            warnings=list(result.warnings),
            risk_score=risk_score,
            risk_level=risk_score,
            repository=repository,
            branch=branch,
            changed_files=changed_files or [],
            patch_size={
                "files_changed": len(changed_files or []),
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
            },
            required_checks=required_checks or [],
            failed_checks=failed_checks or [],
            timestamp=result.evaluated_at,
        )

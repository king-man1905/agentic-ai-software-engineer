---
name: engineering-review
description: Use when reviewing a requirement and implementing a software change in this workspace — a step-by-step engineering workflow from understanding the requirement through to explaining the final diff. Trigger on requests to implement a feature, fix a bug, or review a change for correctness and quality.
---

# Engineering Review

A disciplined workflow for turning a requirement into a correct, reviewed
change. Follow the steps in order; don't skip ahead to implementation before
the plan is clear.

## 1. Understand the requirement and constraints

Restate what is being asked in your own words before touching code. Identify:
- The actual goal, not just the literal request — what problem does this solve?
- Constraints: performance, compatibility, existing conventions, dependencies
  already in use.
- What's explicitly out of scope.

If the requirement is ambiguous in a way that changes the implementation,
ask before proceeding.

## 2. Inspect existing architecture and search for reusable code

Before writing anything new:
- Read the modules the change will touch, and the code that calls them.
- Search the codebase for existing helpers, patterns, or abstractions that
  already solve part of the problem — reuse over reinvention.
- Understand the conventions already in place (naming, error handling,
  structure) so the change fits in rather than standing out.

## 3. Identify the smallest correct implementation

Once the terrain is understood, decide the actual shape of the fix:
- Prefer the change with the fewest moving parts that is still fully correct.
- Favor fixing root causes over patching symptoms — trace shared code paths
  so the fix applies everywhere the problem occurs, not just the one call
  site that was reported.
- Note any trade-offs made and why.

## 4. Check correctness, edge cases, security, and maintainability

Before implementing, pressure-test the plan:
- Correctness: does it handle the normal case and the documented
  requirements fully?
- Edge cases: empty/null inputs, boundary values, concurrent access,
  partial failures.
- Security: input validation at trust boundaries, injection risks, secrets
  handling, auth/authz implications.
- Maintainability: will the next person understand this without you? Does
  it introduce hidden coupling or fragile assumptions?

## 5. Implement only after the plan is clear

Write the change once steps 1-4 have produced a concrete, validated plan.
Avoid exploratory edits that outrun the understanding of the problem.

## 6. Run relevant tests/checks

After implementing:
- Run the existing automated tests that cover the changed code.
- Run linters/type-checkers/build steps the project uses, if applicable.
- If no test covers the change, add or run a minimal check that would fail
  if the logic were wrong.
- Report actual command output — don't claim something passes without
  having run it.

## 7. Review the final diff and explain what changed and why

Before considering the work done:
- Read through the actual diff, not just your memory of the intended change.
- Confirm nothing unrelated slipped in.
- Summarize what changed and why, in terms of the original requirement —
  enough for a reviewer to understand the change without re-deriving it.

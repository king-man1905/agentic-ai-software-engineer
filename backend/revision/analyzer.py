import re
from typing import List, Optional, Set
from backend.revision.models import ParsedFailure, ErrorTraceAnalysis
from backend.sandbox.models import TestExecutionResult


class ErrorTraceAnalyzer:
    """
    Analyzes and extracts structured debugging details from test execution error tracebacks,
    pytest failure blocks, and runtime exceptions.
    """

    @classmethod
    def analyze_test_result(
        cls, test_result: Optional[TestExecutionResult]
    ) -> ErrorTraceAnalysis:
        """
        Extracts and analyzes failure traces directly from a TestExecutionResult.
        """
        if test_result is None:
            return ErrorTraceAnalysis(
                failing_tests=[],
                failures=[],
                error_traceback="",
                diagnosis="No test execution result provided.",
            )

        # Prefer error_summary, fallback to stderr or stdout
        trace_text = (
            test_result.error_summary
            or test_result.stderr
            or test_result.stdout
            or ""
        )

        return cls.analyze(trace_text)

    @classmethod
    def analyze(cls, trace_text: str) -> ErrorTraceAnalysis:
        """
        Parses the provided trace string and returns an aggregated ErrorTraceAnalysis.
        """
        if not trace_text or not trace_text.strip():
            return ErrorTraceAnalysis(
                failing_tests=[],
                failures=[],
                error_traceback="",
                diagnosis="No test failures detected or trace is empty.",
            )

        clean_trace = trace_text.strip()
        failures = cls.parse_failures(clean_trace)

        failing_tests: List[str] = []
        seen_test_ids: Set[str] = set()

        for fail in failures:
            test_id = fail.full_test_id or fail.test_name or fail.test_file
            if test_id and test_id not in seen_test_ids:
                seen_test_ids.add(test_id)
                failing_tests.append(test_id)

        # Build comprehensive diagnostic summary
        diagnosis = cls._build_diagnosis(failures, clean_trace)

        return ErrorTraceAnalysis(
            failing_tests=failing_tests,
            failures=failures,
            error_traceback=clean_trace,
            diagnosis=diagnosis,
        )

    @classmethod
    def _normalize_test_key(cls, test_id: str) -> str:
        if not test_id:
            return ""
        norm = test_id.replace("\\", "/").strip()
        # Replace class dots with :: for matching e.g. TestClass.test_fn -> TestClass::test_fn
        return norm.replace(".", "::")

    @classmethod
    def parse_failures(cls, trace_text: str) -> List[ParsedFailure]:
        """
        Parses all failure blocks within pytest output or standard python tracebacks.
        """
        failures: List[ParsedFailure] = []

        # 1. First attempt: Parse Pytest Section Blocks (e.g., '___ test_foo ___')
        section_failures = cls._parse_pytest_sections(trace_text)
        if section_failures:
            failures.extend(section_failures)

        # 2. Check Pytest Short Summary section if present (e.g., 'FAILED tests/test_foo.py::test_bar - ...')
        summary_failures = cls._parse_pytest_short_summary(trace_text)
        if summary_failures:
            existing_keys = {cls._normalize_test_key(f.full_test_id) for f in failures if f.full_test_id}
            for sf in summary_failures:
                sf_key = cls._normalize_test_key(sf.full_test_id)
                if sf_key not in existing_keys and not any(sf.test_name in ef.full_test_id for ef in failures):
                    failures.append(sf)
                    existing_keys.add(sf_key)
                else:
                    # Enrich existing failure if it was missing error_message or exception
                    for ef in failures:
                        ef_key = cls._normalize_test_key(ef.full_test_id)
                        if ef_key == sf_key or sf.test_name in ef.full_test_id or ef.test_name in sf.full_test_id:
                            if not ef.error_message and sf.error_message:
                                ef.error_message = sf.error_message
                            if not ef.exception_type and sf.exception_type:
                                ef.exception_type = sf.exception_type
                            if sf.test_file and not ef.test_file:
                                ef.test_file = sf.test_file
                            if "::" in sf.full_test_id and "::" not in ef.full_test_id:
                                ef.full_test_id = sf.full_test_id

        # 3. Fallback: Parse standard Python traceback if no pytest sections found
        if not failures and ("Traceback (most recent call last):" in trace_text or "Error:" in trace_text):
            tb_failure = cls._parse_python_traceback(trace_text)
            if tb_failure:
                failures.append(tb_failure)

        return failures

    @classmethod
    def _parse_pytest_sections(cls, trace_text: str) -> List[ParsedFailure]:
        failures: List[ParsedFailure] = []

        # Solid underscores with at least 5 underscores, no spaces in header delimiters
        # Matches: _________________________ test_divide _________________________
        header_pattern = re.compile(r"^_{5,}\s*([A-Za-z0-9].*?)\s*_{5,}$", re.MULTILINE)
        matches = list(header_pattern.finditer(trace_text))

        if not matches:
            return failures

        for i, match in enumerate(matches):
            header_raw = match.group(1).strip()
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(trace_text)
            block = trace_text[start_idx:end_idx].strip()

            # Truncate any trailing summary marker lines (e.g. === short test summary info ===)
            if "===" in block:
                block_parts = re.split(r"^={3,}.*?={3,}$", block, flags=re.MULTILINE)
                if block_parts:
                    block = block_parts[0].strip()

            # Handle Collection Errors vs Test Failures
            if header_raw.startswith("ERROR collecting "):
                test_file = header_raw.replace("ERROR collecting ", "").strip()
                test_name = test_file
                full_test_id = test_file
            else:
                test_name = header_raw
                test_file = ""
                full_test_id = test_name

            # Extract line numbers and file paths from the block
            target_lines: List[int] = []
            file_candidates: List[str] = []

            # Patterns like: tests/test_foo.py:25: in test_bar OR tests/test_foo.py:25: AssertionError
            line_file_matches = re.findall(
                r"([\w\-./\\]+\.py):(\d+)(?::\s*(?:in\s+(\w+)|([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception)))?",
                block,
            )
            for f_path, line_str, fn_name, exc_name in line_file_matches:
                line_num = int(line_str)
                if line_num not in target_lines:
                    target_lines.append(line_num)
                if not test_file and ("test" in f_path.lower() or f_path.endswith(".py")):
                    file_candidates.append(f_path)
                if fn_name and not test_name:
                    test_name = fn_name

            if not test_file and file_candidates:
                test_file = file_candidates[0]

            # Reconstruct full test id if file and name are known
            if test_file and test_name and not full_test_id.startswith(test_file):
                # Replace dots in class with :: for standard pytest test id
                formatted_name = test_name.replace(".", "::") if "." in test_name else test_name
                full_test_id = f"{test_file}::{formatted_name}"

            # Extract exception type and assertion error messages
            # Look for lines starting with 'E   '
            e_lines = []
            for line in block.splitlines():
                if line.startswith("E   "):
                    e_lines.append(line[4:].strip())
                elif line.startswith("E "):
                    e_lines.append(line[2:].strip())

            error_message = "\n".join(e_lines) if e_lines else ""
            exception_type = None

            # Detect exception type
            exc_match = re.search(r"([A-Za-z0-9_]+(?:Error|Exception|AssertionError)):?\s*(.*)", block)
            if exc_match:
                exception_type = exc_match.group(1).strip()
                if not error_message:
                    error_message = exc_match.group(0).strip()
            elif "assert " in error_message or "AssertionError" in block:
                exception_type = "AssertionError"

            failures.append(
                ParsedFailure(
                    test_file=test_file,
                    test_name=test_name,
                    full_test_id=full_test_id,
                    exception_type=exception_type,
                    error_message=error_message,
                    target_lines=target_lines,
                    traceback_snippet=block[:1000],
                )
            )

        return failures

    @classmethod
    def _parse_pytest_short_summary(cls, trace_text: str) -> List[ParsedFailure]:
        failures: List[ParsedFailure] = []

        summary_marker = "short test summary info"
        if summary_marker in trace_text:
            summary_part = trace_text.split(summary_marker, 1)[1]
        else:
            summary_part = trace_text

        # Lines like:
        # FAILED tests/test_calc.py::test_calc_divide - ZeroDivisionError: division by zero
        # ERROR tests/test_init.py - ModuleNotFoundError: No module named 'x'
        pattern = re.compile(
            r"^(?:FAILED|ERROR)\s+([\w\-./\\]+)(?:::([\w\-.:\[\]]+))?\s*(?:-\s*(.*))?$",
            re.MULTILINE,
        )

        for match in pattern.finditer(summary_part):
            test_file = match.group(1).strip()
            test_name = match.group(2).strip() if match.group(2) else test_file
            msg = match.group(3).strip() if match.group(3) else ""

            full_id = f"{test_file}::{test_name}" if test_name != test_file else test_file

            exception_type = None
            if ":" in msg:
                potential_exc = msg.split(":", 1)[0].strip()
                if (
                    "Error" in potential_exc
                    or "Exception" in potential_exc
                    or potential_exc == "assert"
                ):
                    exception_type = potential_exc

            failures.append(
                ParsedFailure(
                    test_file=test_file,
                    test_name=test_name,
                    full_test_id=full_id,
                    exception_type=exception_type,
                    error_message=msg,
                    target_lines=[],
                    traceback_snippet=match.group(0),
                )
            )

        return failures

    @classmethod
    def _parse_python_traceback(cls, trace_text: str) -> Optional[ParsedFailure]:
        target_lines: List[int] = []
        files: List[str] = []
        functions: List[str] = []

        # File "...", line 12, in function_name
        tb_matches = re.findall(
            r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?', trace_text
        )
        for f_path, line_str, fn_name in tb_matches:
            target_lines.append(int(line_str))
            files.append(f_path)
            if fn_name:
                functions.append(fn_name)

        # Extract last exception line
        exc_match = re.search(
            r"([A-Za-z0-9_.]+(?:Error|Exception)):\s*(.*)", trace_text
        )
        exception_type = exc_match.group(1).strip() if exc_match else None
        error_message = exc_match.group(0).strip() if exc_match else ""

        test_file = files[-1] if files else ""
        test_name = functions[-1] if functions else "execution_error"
        full_test_id = f"{test_file}::{test_name}" if test_file else test_name

        return ParsedFailure(
            test_file=test_file,
            test_name=test_name,
            full_test_id=full_test_id,
            exception_type=exception_type,
            error_message=error_message,
            target_lines=target_lines,
            traceback_snippet=trace_text[:1000],
        )

    @classmethod
    def _build_diagnosis(
        cls, failures: List[ParsedFailure], clean_trace: str
    ) -> str:
        if not failures:
            lines = clean_trace.splitlines()
            short_preview = "\n".join(lines[:10])
            return f"Execution failure without structured test cases:\n{short_preview}"

        lines = [f"Identified {len(failures)} failing test case(s):"]
        for idx, f in enumerate(failures, 1):
            loc_str = ""
            if f.test_file:
                loc_str += f" in `{f.test_file}`"
            if f.target_lines:
                lines_str = ", ".join(str(l) for l in f.target_lines)
                loc_str += f" (line(s): {lines_str})"

            lines.append(f"{idx}. Test: `{f.full_test_id or f.test_name}`{loc_str}")
            if f.exception_type:
                lines.append(f"   Root Exception: {f.exception_type}")
            if f.error_message:
                # Format multi-line error messages with indentation
                formatted_msg = "\n   ".join(f.error_message.splitlines())
                lines.append(f"   Error Details: {formatted_msg}")

        return "\n".join(lines)


def analyze_error_trace(trace_text: str) -> ErrorTraceAnalysis:
    """
    Functional helper to analyze an error trace string.
    """
    return ErrorTraceAnalyzer.analyze(trace_text)

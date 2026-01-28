from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 11)
MIN_UV = (0, 9, 25)


def parse_semver(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def check_python() -> tuple[bool, str]:
    current = sys.version_info[:3]
    ok = current >= MIN_PYTHON
    return (
        ok,
        f"Python {current[0]}.{current[1]}.{current[2]} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    )


def check_uv() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "uv not found in PATH"
    except subprocess.CalledProcessError as exc:
        return False, f"uv version check failed: {exc}"

    version_text = result.stdout.strip() or result.stderr.strip()
    parsed = parse_semver(version_text)
    if not parsed:
        return False, f"uv version parse failed: '{version_text}'"

    ok = parsed >= MIN_UV
    return (
        ok,
        f"uv {parsed[0]}.{parsed[1]}.{parsed[2]} (>= {MIN_UV[0]}.{MIN_UV[1]}.{MIN_UV[2]})",
    )


def check_exists(path: Path, label: str) -> tuple[bool, str]:
    if path.exists():
        return True, f"{label} exists"
    return False, f"{label} missing: {path}"


def run_checks(project_root: Path) -> int:
    checks: list[tuple[str, bool, str]] = []

    ok, detail = check_python()
    checks.append(("python", ok, detail))

    ok, detail = check_uv()
    checks.append(("uv", ok, detail))

    ok, detail = check_exists(project_root / "pyproject.toml", "pyproject.toml")
    checks.append(("pyproject", ok, detail))

    ok, detail = check_exists(project_root / "uv.lock", "uv.lock")
    checks.append(("uv_lock", ok, detail))

    ok, detail = check_exists(project_root / "packages", "packages")
    checks.append(("packages", ok, detail))

    failures = 0
    for _name, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        print(f"[{status}] {detail}")
        if not passed:
            failures += 1

    passed_count = len(checks) - failures
    print(f"Summary: {passed_count} passed, {failures} failed")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="프로젝트 초기 설정 상태를 점검합니다."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="프로젝트 루트 경로 (기본값: 스크립트 위치 기준)",
    )
    args = parser.parse_args()

    if args.project_root:
        root = Path(args.project_root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[1]

    return run_checks(root)


if __name__ == "__main__":
    raise SystemExit(main())

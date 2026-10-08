"""Check PR_TITLE using the repository's documented pull-request title format."""

import os
import re
import sys

TITLE_PATTERN = re.compile(
    r"(?:feat|fix|refactor|perf|docs|test|build|ci|chore)"
    r"(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S[^\r\n]*"
)


def main() -> int:
    if TITLE_PATTERN.fullmatch(os.environ.get("PR_TITLE", "")):
        print("PR title format is valid.")
        return 0
    print("Use type(scope): description, e.g. fix(gui): 保留工程保存路径", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

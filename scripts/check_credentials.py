"""Print credential presence only. Does not authenticate or make any network request."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

NAMES = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    env = dict(os.environ)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("env file not found")
        for line in args.env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                parser.error("expected NAME=value; values are never printed")
            key, value = line.split("=", 1)
            if key.strip() not in (*NAMES, "VIDEO_REPORT_VIDEO_ROOT"):
                continue
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                parser.error("invalid quoting; values are never printed")
            if len(parts) > 1:
                parser.error("quote values containing spaces")
            env[key.strip()] = parts[0] if parts else ""
    missing = False
    for name in NAMES:
        present = bool(env.get(name, "").strip())
        print(f"{name}: {'present (not authenticated)' if present else 'missing'}")
        missing = missing or not present
    print("Local presence check only. No API requests and no credits used.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

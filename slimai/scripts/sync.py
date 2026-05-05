#!/usr/bin/env python3
from pathlib import Path
import sys


def _prepare_env() -> None:
  package_dir = Path(__file__).resolve().parents[1]
  project_dir = package_dir.parent
  project_path = project_dir.as_posix()
  if project_path not in sys.path:
    sys.path.insert(0, project_path)
  return


def main() -> int:
  _prepare_env()
  from slimai.helper.features.copy import main as copy_main

  return int(copy_main())


if __name__ == "__main__":
  if len(sys.argv) == 1:
    print(
      "Usage example:\n"
      "  python3 slimai/slimai/scripts/sync.py sync "
      "--xlsx /path/to/input.xlsx --path-column EMBEDDING "
      "--src-base /path/to/src --dst /path/to/dst"
    )
  raise SystemExit(main())

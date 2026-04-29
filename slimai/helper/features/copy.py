#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence, Tuple

# Avoid shadowing stdlib `copy` when this file is executed directly.
if __name__ == "__main__":
  _this_dir = Path(__file__).resolve().parent
  sys.path = [p for p in sys.path if Path(p or ".").resolve() != _this_dir]

from dataclasses import dataclass

import pandas as pd
import torch
from tqdm import tqdm


@dataclass
class CliDefaults:
  excel: Optional[str] = None
  sheet: Optional[str] = None
  column: Optional[str] = None
  old_prefix: Optional[str] = None
  new_prefix: Optional[str] = None
  target_root: Optional[str] = None
  valid_root: Optional[str] = None
  log_file: Optional[str] = None
  verified_file_name: Optional[str] = None
  workers: Optional[int] = None


@dataclass
class ParseResult:
  total_rows: int
  empty_rows: int
  not_pkl_rows: int
  wrong_prefix_rows: int
  valid_unique_count: int
  duplicate_count: int
  relative_paths: List[str]


def map_path(path: str, old_prefix: str, new_prefix: str) -> str:
  if path.startswith(old_prefix):
    return new_prefix + path[len(old_prefix) :]
  return path.replace(old_prefix, new_prefix, 1)


def collect_relative_paths(
  df: pd.DataFrame,
  column: str,
  old_prefix: str,
  new_prefix: str,
) -> Tuple[pd.DataFrame, ParseResult]:
  if column not in df.columns:
    raise KeyError(f"Column '{column}' does not exist in sheet.")

  total_rows = int(len(df))
  series = df[column]
  mapped_paths: List[Optional[str]] = []
  relative_paths: List[str] = []

  empty_rows = 0
  not_pkl_rows = 0
  wrong_prefix_rows = 0
  duplicate_count = 0
  seen = set()

  for value in series.tolist():
    if pd.isna(value):
      empty_rows += 1
      mapped_paths.append(None)
      continue

    raw_path = str(value).strip()
    if not raw_path:
      empty_rows += 1
      mapped_paths.append(None)
      continue

    mapped = map_path(raw_path, old_prefix=old_prefix, new_prefix=new_prefix)
    mapped_paths.append(mapped)

    if not mapped.endswith(".pkl"):
      not_pkl_rows += 1
      continue
    if not mapped.startswith(new_prefix):
      wrong_prefix_rows += 1
      continue

    relative = mapped[len(new_prefix) :].lstrip("/")
    if not relative:
      wrong_prefix_rows += 1
      continue

    if relative in seen:
      duplicate_count += 1
      continue
    seen.add(relative)
    relative_paths.append(relative)

  out_df = df.copy()
  out_df[column] = mapped_paths
  result = ParseResult(
    total_rows=total_rows,
    empty_rows=empty_rows,
    not_pkl_rows=not_pkl_rows,
    wrong_prefix_rows=wrong_prefix_rows,
    valid_unique_count=len(relative_paths),
    duplicate_count=duplicate_count,
    relative_paths=relative_paths,
  )
  return out_df, result


def write_excel_copy(
  input_excel: Path,
  output_excel: Path,
  target_sheet: str,
  output_sheet_df: pd.DataFrame,
) -> None:
  all_sheets = pd.read_excel(input_excel, sheet_name=None)
  if target_sheet not in all_sheets:
    raise KeyError(f"Sheet '{target_sheet}' does not exist in workbook.")
  all_sheets[target_sheet] = output_sheet_df
  output_excel.parent.mkdir(parents=True, exist_ok=True)
  with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    for sheet_name, sheet_df in all_sheets.items():
      sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_manifest_bytes(relative_paths: Sequence[str], manifest_path: Path) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  with manifest_path.open("wb") as f:
    for relative in relative_paths:
      f.write(relative.encode("utf-8"))
      f.write(b"\0")


def _format_bytes(size: int) -> str:
  units = ["B", "KB", "MB", "GB", "TB"]
  value = float(size)
  for unit in units:
    if value < 1024.0 or unit == units[-1]:
      return f"{value:.2f}{unit}"
    value /= 1024.0
  return f"{size}B"


def _format_bandwidth(bytes_count: int, elapsed_seconds: float) -> str:
  if elapsed_seconds <= 0:
    return "0.00MB/s"
  mb_per_second = bytes_count / elapsed_seconds / 1024 / 1024
  return f"{mb_per_second:.2f}MB/s"


def _log(log_fp, message: str) -> None:
  print(message)
  log_fp.write(message + "\n")
  log_fp.flush()


def fsync_file_object(file_obj) -> None:
  file_obj.flush()
  try:
    os.fsync(file_obj.fileno())
  except OSError:
    return


def fsync_path_file(path: Path) -> None:
  if not path.exists() or not path.is_file():
    return
  try:
    with path.open("rb") as f:
      os.fsync(f.fileno())
  except OSError:
    return


def fsync_parent_dir(path: Path) -> None:
  parent = path.parent
  flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
  try:
    fd = os.open(parent.as_posix(), flags)
  except OSError:
    return
  try:
    os.fsync(fd)
  except OSError:
    pass
  finally:
    os.close(fd)


def atomic_unlink(path: Path) -> None:
  if not path.exists():
    return
  path.unlink()
  fsync_parent_dir(path)


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.md5()
  with path.open("rb") as f:
    while True:
      chunk = f.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def load_verified_records(verified_file: Path, missing_ok: bool = True) -> dict:
  if not verified_file.exists():
    if missing_ok:
      return {}
    raise FileNotFoundError(f"Verified file not found: {verified_file}")

  records = {}
  with verified_file.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        event = json.loads(line)
      except Exception:
        continue
      relative = event.get("relative", None)
      status = event.get("status", None)
      if not isinstance(relative, str):
        continue
      if status == "delete":
        records.pop(relative, None)
        continue
      if status == "verified":
        record = event.get("record", None)
        if isinstance(record, dict):
          records[relative] = record
  return records


def append_verified_event(
  verified_file: Path,
  *,
  relative: str,
  status: str,
  record: Optional[dict] = None,
) -> None:
  verified_file.parent.mkdir(parents=True, exist_ok=True)
  event: dict = dict(
    relative=relative,
    status=status,
    ts=datetime.now().isoformat(),
  )
  if record is not None:
    event["record"] = record
  with verified_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    fsync_file_object(f)


def is_verified_record_valid(
  record: dict,
  source_path: Path,
  target_path: Path,
  verify_md5: bool,
) -> bool:
  if not target_path.exists():
    return False
  source_stat = source_path.stat()
  target_stat = target_path.stat()
  source_size = int(record.get("source_size", -1))
  target_size = int(record.get("target_size", -2))
  size_valid = (
    int(source_stat.st_size) == source_size
    and source_size == target_size
    and int(target_stat.st_size) == target_size
  )
  if not size_valid:
    return False
  if not verify_md5:
    return True

  source_md5 = record.get("source_md5", None)
  target_md5 = record.get("target_md5", None)
  return (
    source_md5 is not None
    and target_md5 is not None
    and source_md5 == target_md5
  )


def build_verified_record(
  source_path: Path,
  target_path: Path,
  source_md5: Optional[str] = None,
  target_md5: Optional[str] = None,
) -> dict:
  source_stat = source_path.stat()
  target_stat = target_path.stat()
  record = dict(
    source_size=int(source_stat.st_size),
    target_size=int(target_stat.st_size),
    verify_mode="size+md5" if source_md5 is not None and target_md5 is not None else "size",
    verified_at=datetime.now().isoformat(),
  )
  if source_md5 is not None and target_md5 is not None:
    record["source_md5"] = source_md5
    record["target_md5"] = target_md5
  return record


def process_single_file(
  *,
  index: int,
  total_files: int,
  relative: str,
  source_root: Path,
  target_root: Path,
  verified_record: Optional[dict],
  verify_md5: bool,
) -> dict:
  progress_head = f"[{index}/{total_files}]"
  source_path = source_root / relative
  target_path = target_root / relative
  partial_path = target_path.with_suffix(target_path.suffix + ".partial")

  counters = dict(
    done_files=0,
    skipped_files=0,
    skip_verified_files=0,
    verified_files=0,
    resumed_files=0,
    copied_files=0,
    repaired_files=0,
    retry_files=0,
    deleted_bad_files=0,
    md5_mismatch_files=0,
    missing_files=0,
    failed_files=0,
    completed_bytes=0,
    copied_bytes=0,
  )
  events: List[str] = []
  verified_update = None

  if not source_path.exists():
    counters["missing_files"] += 1
    verified_update = "__DELETE__"
    events.append(f"{progress_head} MISSING {source_path.as_posix()}")
    return dict(relative=relative, counters=counters, events=events, verified_update=verified_update)

  source_size = source_path.stat().st_size
  source_md5 = None

  def get_source_md5():
    nonlocal source_md5
    if source_md5 is None:
      source_md5 = file_md5(source_path)
    return source_md5

  if verified_record is not None:
    if is_verified_record_valid(
      verified_record,
      source_path=source_path,
      target_path=target_path,
      verify_md5=verify_md5,
    ):
      counters["skip_verified_files"] += 1
      counters["done_files"] += 1
      counters["completed_bytes"] += source_size
      events.append(
        f"{progress_head} SKIP_VERIFIED {relative} ({_format_bytes(source_size)})"
      )
      return dict(relative=relative, counters=counters, events=events, verified_update=None)
    verified_update = "__DELETE__"

  try:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_valid = False
    target_md5 = None
    if target_path.exists():
      current_size = target_path.stat().st_size
      if current_size == source_size:
        if not verify_md5:
          fsync_path_file(target_path)
          target_valid = True
        else:
          target_md5 = file_md5(target_path)
          if target_md5 == get_source_md5():
            fsync_path_file(target_path)
            target_valid = True
          else:
            counters["md5_mismatch_files"] += 1
            events.append(
              f"{progress_head} REPAIR_MD5_MISMATCH {relative} "
              f"src_md5={get_source_md5()} dst_md5={target_md5}"
            )
      else:
        events.append(
          f"{progress_head} REPAIR_SIZE_MISMATCH {relative} "
          f"src={source_size} dst={current_size}"
        )

    if target_valid:
      counters["skipped_files"] += 1
      counters["verified_files"] += 1
      counters["done_files"] += 1
      counters["completed_bytes"] += source_size
      verified_update = build_verified_record(
        source_path=source_path,
        target_path=target_path,
        source_md5=get_source_md5() if verify_md5 else None,
        target_md5=target_md5,
      )
      events.append(f"{progress_head} SKIP {relative} ({_format_bytes(source_size)})")
      return dict(relative=relative, counters=counters, events=events, verified_update=verified_update)

    if target_path.exists():
      atomic_unlink(target_path)
    if partial_path.exists() and partial_path.stat().st_size > source_size:
      atomic_unlink(partial_path)

    copied_success = False
    repaired_flag = False
    last_action = "COPY"
    for attempt in range(2):
      if attempt == 1:
        counters["retry_files"] += 1
        repaired_flag = True
        events.append(f"{progress_head} RETRY_COPY {relative} attempt=2")
        if partial_path.exists():
          atomic_unlink(partial_path)

      partial_exists = partial_path.exists()
      partial_size = partial_path.stat().st_size if partial_exists else 0

      if partial_exists and partial_size == source_size:
        if not verify_md5:
          partial_path.replace(target_path)
          fsync_parent_dir(target_path)
          fsync_path_file(target_path)
          last_action = "PUBLISH"
        else:
          partial_md5 = file_md5(partial_path)
          if partial_md5 == get_source_md5():
            partial_path.replace(target_path)
            fsync_parent_dir(target_path)
            fsync_path_file(target_path)
            last_action = "PUBLISH"
          else:
            counters["md5_mismatch_files"] += 1
            events.append(
              f"{progress_head} PARTIAL_MD5_MISMATCH {relative} "
              f"src_md5={get_source_md5()} partial_md5={partial_md5}"
            )
            atomic_unlink(partial_path)
            partial_exists = False
            partial_size = 0

      if not target_path.exists():
        if partial_exists and partial_size < source_size:
          counters["resumed_files"] += 1
          copied_now = source_size - partial_size
          counters["copied_bytes"] += copied_now
          with source_path.open("rb") as src, partial_path.open("ab") as dst:
            src.seek(partial_size)
            while True:
              chunk = src.read(8 * 1024 * 1024)
              if not chunk:
                break
              dst.write(chunk)
            fsync_file_object(dst)
          last_action = f"RESUME +{_format_bytes(copied_now)}"
        elif not partial_exists:
          counters["copied_files"] += 1
          counters["copied_bytes"] += source_size
          with source_path.open("rb") as src, partial_path.open("wb") as dst:
            while True:
              chunk = src.read(8 * 1024 * 1024)
              if not chunk:
                break
              dst.write(chunk)
            fsync_file_object(dst)
          last_action = f"COPY {_format_bytes(source_size)}"

        if partial_path.exists():
          new_size = partial_path.stat().st_size
          if new_size != source_size:
            events.append(
              f"{progress_head} FAIL_SIZE_MISMATCH {relative} "
              f"src={source_size} partial={new_size} attempt={attempt + 1}"
            )
            continue
          partial_md5 = file_md5(partial_path)
          if partial_md5 != get_source_md5():
            counters["md5_mismatch_files"] += 1
            events.append(
              f"{progress_head} FAIL_MD5_MISMATCH {relative} "
              f"src_md5={get_source_md5()} partial_md5={partial_md5} attempt={attempt + 1}"
            )
            continue
          partial_path.replace(target_path)
          fsync_parent_dir(target_path)
          fsync_path_file(target_path)

      final_size = target_path.stat().st_size if target_path.exists() else -1
      if final_size != source_size:
        events.append(
          f"{progress_head} FAIL_FINAL_SIZE {relative} "
          f"src={source_size} dst={final_size} attempt={attempt + 1}"
        )
        continue
      final_md5 = file_md5(target_path)
      if final_md5 != get_source_md5():
        counters["md5_mismatch_files"] += 1
        events.append(
          f"{progress_head} FAIL_FINAL_MD5 {relative} "
          f"src_md5={get_source_md5()} dst_md5={final_md5} attempt={attempt + 1}"
        )
        if target_path.exists():
          atomic_unlink(target_path)
        continue

      verified_update = build_verified_record(
        source_path=source_path,
        target_path=target_path,
        source_md5=get_source_md5(),
        target_md5=final_md5,
      )
      counters["verified_files"] += 1
      counters["done_files"] += 1
      counters["completed_bytes"] += source_size
      if repaired_flag:
        counters["repaired_files"] += 1
      copied_success = True
      events.append(f"{progress_head} {last_action} {relative}")
      break

    if not copied_success:
      counters["failed_files"] += 1
      counters["deleted_bad_files"] += 1
      if partial_path.exists():
        atomic_unlink(partial_path)
      if target_path.exists():
        atomic_unlink(target_path)
      verified_update = "__DELETE__"
      events.append(
        f"{progress_head} FAIL_VERIFY_DELETE {relative} after two attempts (copy + one retry)"
      )
  except Exception as ex:
    counters["failed_files"] += 1
    events.append(f"{progress_head} FAIL {relative}: {ex}")

  return dict(relative=relative, counters=counters, events=events, verified_update=verified_update)


def count_cached_pkl_files(target_root: Path) -> int:
  if not target_root.exists():
    return 0
  return sum(1 for _ in target_root.rglob("*.pkl"))


def parse_latest_python_summary(log_file: Path) -> Optional[dict]:
  if not log_file.exists():
    return None

  lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
  marker = "---------- Python copy summary ----------"
  last_idx = -1
  for idx, line in enumerate(lines):
    if marker in line:
      last_idx = idx
  if last_idx < 0:
    return None

  summary = {}
  for line in lines[last_idx + 1 :]:
    if "Python copy fallback finished" in line:
      break
    match = re.match(r"^([a-z_]+):\s*(.+)$", line.strip())
    if match:
      summary[match.group(1)] = match.group(2)

  if not summary:
    return None
  return summary


def show_status(target_root: Path, log_file: Path) -> int:
  print("========== CACHE STATUS ==========")
  print(f"target root: {target_root.as_posix()}")
  print(f"log file: {log_file.as_posix()}")
  print(f"cached_pkl_files: {count_cached_pkl_files(target_root)}")

  latest_summary = parse_latest_python_summary(log_file)
  if latest_summary is None:
    print("latest_python_summary: not found")
    return 0

  print("latest_python_summary:")
  keys = [
    "done_files",
    "copied_files",
    "resumed_files",
    "skipped_files",
    "skip_verified_files",
    "verified_files",
    "repaired_files",
    "retry_files",
    "deleted_bad_files",
    "md5_mismatch_files",
    "missing_files",
    "failed_files",
    "completed_bytes",
    "copied_bytes",
    "avg_copy_bandwidth",
  ]
  for key in keys:
    if key in latest_summary:
      print(f"  {key}: {latest_summary[key]}")
  return 0


def copy_with_python_resume(
  relative_paths: Sequence[str],
  source_root: Path,
  target_root: Path,
  dry_run: bool,
  log_file: Path,
  verified_file: Path,
  workers: int,
  verify_md5: bool,
) -> int:
  total_files = len(relative_paths)
  done_files = 0
  skipped_files = 0
  skip_verified_files = 0
  verified_files = 0
  resumed_files = 0
  copied_files = 0
  repaired_files = 0
  retry_files = 0
  deleted_bad_files = 0
  md5_mismatch_files = 0
  missing_files = 0
  failed_files = 0
  total_bytes = 0
  completed_bytes = 0
  copied_bytes = 0
  verified_records = load_verified_records(verified_file=verified_file, missing_ok=True)
  start_time = time.monotonic()

  if not dry_run:
    for relative in relative_paths:
      source_path = source_root / relative
      if source_path.exists():
        total_bytes += source_path.stat().st_size

  log_file.parent.mkdir(parents=True, exist_ok=True)
  with log_file.open("a", encoding="utf-8") as lf:
    _log(lf, f"\n[{datetime.now().isoformat()}] Python copy fallback (rsync unavailable)")
    _log(lf, f"total files in manifest: {total_files}")
    _log(lf, f"workers: {workers}")
    _log(lf, f"verify existing cached files with md5: {verify_md5}")
    _log(lf, "copied files verification: size+md5")
    if dry_run:
      _log(lf, "total source bytes: N/A (dry-run skips remote stat)")
      _log(lf, "mode: dry-run (no file transfer)")
    else:
      _log(lf, f"total source bytes (existing only): {_format_bytes(total_bytes)}")
      _log(lf, "mode: real copy")

    def apply_result(result: dict) -> None:
      nonlocal done_files, skipped_files, skip_verified_files, verified_files
      nonlocal resumed_files, copied_files, repaired_files, retry_files
      nonlocal deleted_bad_files, md5_mismatch_files, missing_files
      nonlocal failed_files, completed_bytes, copied_bytes

      counters = result["counters"]
      done_files += counters["done_files"]
      skipped_files += counters["skipped_files"]
      skip_verified_files += counters["skip_verified_files"]
      verified_files += counters["verified_files"]
      resumed_files += counters["resumed_files"]
      copied_files += counters["copied_files"]
      repaired_files += counters["repaired_files"]
      retry_files += counters["retry_files"]
      deleted_bad_files += counters["deleted_bad_files"]
      md5_mismatch_files += counters["md5_mismatch_files"]
      missing_files += counters["missing_files"]
      failed_files += counters["failed_files"]
      completed_bytes += counters["completed_bytes"]
      copied_bytes += counters["copied_bytes"]

      for event in result["events"]:
        lf.write(event + "\n")
      fsync_file_object(lf)

      verified_update = result.get("verified_update", None)
      relative = result.get("relative", None)
      if relative is not None and verified_update == "__DELETE__":
        if relative in verified_records:
          verified_records.pop(relative, None)
        append_verified_event(
          verified_file=verified_file,
          relative=relative,
          status="delete",
        )
      elif relative is not None and isinstance(verified_update, dict):
        verified_records[relative] = verified_update
        append_verified_event(
          verified_file=verified_file,
          relative=relative,
          status="verified",
          record=verified_update,
        )

    if dry_run:
      with tqdm(total=total_files, desc="Verifying/copying pkl", unit="file") as pbar:
        for index, relative in enumerate(relative_paths, start=1):
          result = dict(
            relative=relative,
            counters=dict(
              done_files=1,
              skipped_files=0,
              skip_verified_files=0,
              verified_files=0,
              resumed_files=0,
              copied_files=0,
              repaired_files=0,
              retry_files=0,
              deleted_bad_files=0,
              md5_mismatch_files=0,
              missing_files=0,
              failed_files=0,
              completed_bytes=0,
              copied_bytes=0,
            ),
            events=[
              f"[{index}/{total_files}] DRY_RUN PLAN {relative} "
              f"file_progress={index}/{total_files}"
            ],
            verified_update=None,
          )
          apply_result(result)
          pbar.update(1)
          pbar.set_postfix(
            copy_bw="N/A",
            verified=verified_files,
            skip_verified=skip_verified_files,
            copied=copied_files,
            repaired=repaired_files,
            retry=retry_files,
            failed=failed_files,
          )
    else:
      with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index, relative in enumerate(relative_paths, start=1):
          futures.append(
            executor.submit(
              process_single_file,
              index=index,
              total_files=total_files,
              relative=relative,
              source_root=source_root,
              target_root=target_root,
              verified_record=verified_records.get(relative, None),
              verify_md5=verify_md5,
            )
          )
        with tqdm(total=total_files, desc="Verifying/copying pkl", unit="file") as pbar:
          for future in as_completed(futures):
            result = future.result()
            apply_result(result)
            pbar.update(1)
            elapsed = time.monotonic() - start_time
            pbar.set_postfix(
              copy_bw=_format_bandwidth(copied_bytes, elapsed),
              verified=verified_files,
              skip_verified=skip_verified_files,
              copied=copied_files,
              repaired=repaired_files,
              retry=retry_files,
              failed=failed_files,
            )

    _log(lf, "---------- Python copy summary ----------")
    _log(lf, f"done_files: {done_files}/{total_files}")
    _log(lf, f"skipped_files: {skipped_files}")
    _log(lf, f"skip_verified_files: {skip_verified_files}")
    _log(lf, f"verified_files: {verified_files}")
    _log(lf, f"resumed_files: {resumed_files}")
    _log(lf, f"copied_files: {copied_files}")
    _log(lf, f"repaired_files: {repaired_files}")
    _log(lf, f"retry_files: {retry_files}")
    _log(lf, f"deleted_bad_files: {deleted_bad_files}")
    _log(lf, f"md5_mismatch_files: {md5_mismatch_files}")
    _log(lf, f"missing_files: {missing_files}")
    _log(lf, f"failed_files: {failed_files}")
    if dry_run:
      _log(lf, "completed_bytes: N/A (dry-run)")
    else:
      _log(
        lf,
        f"completed_bytes: {_format_bytes(completed_bytes)}/{_format_bytes(total_bytes)}",
      )
    elapsed = time.monotonic() - start_time
    _log(lf, f"copied_bytes: {_format_bytes(copied_bytes)}")
    _log(lf, f"avg_copy_bandwidth: {_format_bandwidth(copied_bytes, elapsed)}")
    _log(lf, f"[{datetime.now().isoformat()}] Python copy fallback finished")

  if failed_files > 0:
    return 2
  return 0


def print_summary(result: ParseResult, dry_run: bool, log_file: Path, target_root: Path) -> None:
  print("\n========== SUMMARY ==========")
  print(f"total rows: {result.total_rows}")
  print(f"empty EMBEDDING rows: {result.empty_rows}")
  print(f"non-pkl rows: {result.not_pkl_rows}")
  print(f"prefix-mismatch rows: {result.wrong_prefix_rows}")
  print(f"duplicate relative-path rows skipped: {result.duplicate_count}")
  print(f"valid unique pkl paths: {result.valid_unique_count}")
  print(f"target root: {target_root.as_posix()}")
  print(f"log file: {log_file.as_posix()}")
  if dry_run:
    print("mode: dry-run (no file transfer)")
  else:
    print("mode: real copy")


def is_safe_relative_path(relative: str) -> bool:
  path = Path(relative)
  return (
    relative.endswith(".pkl")
    and not path.is_absolute()
    and ".." not in path.parts
  )


def expected_target_size(record: dict) -> Optional[int]:
  raw_size = record.get("target_size", None)
  if raw_size is None:
    return None
  try:
    return int(raw_size)
  except Exception:
    return None


def is_size_matched_cached_file(source_path: Path, record: dict) -> bool:
  if not source_path.exists() or not source_path.is_file():
    return False
  target_size = expected_target_size(record)
  if target_size is None:
    return False
  return int(source_path.stat().st_size) == target_size


def can_load_with_torch(source_path: Path) -> Tuple[bool, str]:
  try:
    payload = torch.load(
      source_path.as_posix(),
      map_location="cpu",
      weights_only=False,
    )
    del payload
  except Exception as ex:
    return False, str(ex)
  return True, ""


def link_points_to(link_path: Path, source_path: Path) -> bool:
  if not link_path.is_symlink():
    return False
  try:
    return link_path.resolve(strict=False) == source_path.resolve(strict=False)
  except OSError:
    return False


def create_valid_symlinks(
  *,
  records: dict,
  cache_root: Path,
  valid_root: Path,
  dry_run: bool,
  replace: bool,
) -> int:
  counters = dict(
    records=len(records),
    linked=0,
    already_linked=0,
    conflicts=0,
    unsafe_paths=0,
    missing_or_invalid=0,
    load_failed=0,
  )
  load_errors = []

  if not dry_run:
    valid_root.mkdir(parents=True, exist_ok=True)

  for relative, record in tqdm(records.items(), desc="Linking valid pkl", unit="file"):
    if not is_safe_relative_path(relative):
      counters["unsafe_paths"] += 1
      continue

    source_path = cache_root / relative
    link_path = valid_root / relative

    if not is_size_matched_cached_file(source_path=source_path, record=record):
      counters["missing_or_invalid"] += 1
      continue

    load_ok, load_error = can_load_with_torch(source_path)
    if not load_ok:
      counters["load_failed"] += 1
      if len(load_errors) < 10:
        load_errors.append(f"{relative}: {load_error}")
      continue

    if link_points_to(link_path=link_path, source_path=source_path):
      counters["already_linked"] += 1
      continue

    if link_path.exists() or link_path.is_symlink():
      if replace and link_path.is_symlink():
        if not dry_run:
          link_path.unlink()
      else:
        counters["conflicts"] += 1
        continue

    counters["linked"] += 1
    if dry_run:
      continue

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(source_path)

  print("========== LINK SUMMARY ==========")
  print(f"records: {counters['records']}")
  print(f"linked: {counters['linked']}")
  print(f"already_linked: {counters['already_linked']}")
  print(f"conflicts: {counters['conflicts']}")
  print(f"unsafe_paths: {counters['unsafe_paths']}")
  print(f"missing_or_invalid: {counters['missing_or_invalid']}")
  print(f"load_failed: {counters['load_failed']}")
  print(f"cache_root: {cache_root.as_posix()}")
  print(f"valid_root: {valid_root.as_posix()}")
  print(f"mode: {'dry-run' if dry_run else 'real link'}")
  if load_errors:
    print("first_load_errors:")
    for error in load_errors:
      print(f"  {error}")

  if counters["conflicts"] > 0 or counters["load_failed"] > 0:
    return 2
  return 0


def run_copy(args: argparse.Namespace) -> int:
  if args.workers < 1:
    raise ValueError(f"--workers must be >= 1, got {args.workers}")

  excel_path = Path(args.excel)
  source_root = Path(args.new_prefix)
  target_root = Path(args.target_root)
  log_file = Path(args.log_file)
  verified_file = target_root / args.verified_file_name

  if args.status_only:
    return show_status(target_root=target_root, log_file=log_file)

  if not excel_path.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_path}")
  if not source_root.exists():
    raise FileNotFoundError(f"Mapped source root not found: {source_root}")

  target_root.mkdir(parents=True, exist_ok=True)

  df = pd.read_excel(excel_path, sheet_name=args.sheet)
  mapped_df, parse_result = collect_relative_paths(
    df=df,
    column=args.column,
    old_prefix=args.old_prefix,
    new_prefix=args.new_prefix,
  )
  if parse_result.valid_unique_count == 0:
    raise RuntimeError("No valid pkl paths found after mapping. Nothing to copy.")

  if args.write_excel:
    output_excel = Path(args.write_excel)
    write_excel_copy(
      input_excel=excel_path,
      output_excel=output_excel,
      target_sheet=args.sheet,
      output_sheet_df=mapped_df,
    )
    print(f"Mapped Excel has been written to: {output_excel.as_posix()}")

  temp_manifest = None
  if args.manifest:
    manifest_path = Path(args.manifest)
  else:
    temp_manifest = NamedTemporaryFile(
      mode="wb",
      suffix=".manifest",
      prefix="cache_shzs_embedding_",
      delete=False,
    )
    temp_manifest.close()
    manifest_path = Path(temp_manifest.name)

  try:
    write_manifest_bytes(parse_result.relative_paths, manifest_path=manifest_path)
    print(f"Manifest prepared: {manifest_path.as_posix()}")
    print(f"Unique pkl entries: {parse_result.valid_unique_count}")

    print("Running Python strict per-file copy flow (verify -> copy/repair -> verify)...")
    return_code = copy_with_python_resume(
      relative_paths=parse_result.relative_paths,
      source_root=source_root,
      target_root=target_root,
      dry_run=args.dry_run,
      log_file=log_file,
      verified_file=verified_file,
      workers=args.workers,
      verify_md5=args.verify_md5,
    )
  finally:
    if temp_manifest is not None:
      manifest_path.unlink(missing_ok=True)

  print_summary(
    result=parse_result,
    dry_run=args.dry_run,
    log_file=log_file,
    target_root=target_root,
  )
  if return_code != 0:
    print(f"copy flow failed with exit code: {return_code}")
    return return_code
  return 0


def run_link_valid(args: argparse.Namespace) -> int:
  cache_root = Path(args.cache_root)
  valid_root = Path(args.valid_root)
  verified_file = (
    Path(args.verified_file)
    if args.verified_file is not None
    else cache_root / args.verified_file_name
  )
  records = load_verified_records(verified_file=verified_file, missing_ok=False)
  return create_valid_symlinks(
    records=records,
    cache_root=cache_root,
    valid_root=valid_root,
    dry_run=args.dry_run,
    replace=args.replace,
  )


def run_status(args: argparse.Namespace) -> int:
  return show_status(target_root=Path(args.target_root), log_file=Path(args.log_file))


def build_parser(defaults: Optional[CliDefaults] = None) -> argparse.ArgumentParser:
  defaults = defaults or CliDefaults()

  parser = argparse.ArgumentParser(
    description=(
      "Unified embedding cache utility: copy/cache, link valid cache files, and show status."
    )
  )
  subparsers = parser.add_subparsers(dest="command")

  copy_parser = subparsers.add_parser(
    "copy",
    help="Copy embedding pkl files into cache with strict verification.",
    description=(
      "Map EMBEDDING paths from old prefix to new prefix and copy pkl files "
      "to target cache directory with resumable strict verification."
    ),
  )
  copy_parser.add_argument(
    "--excel",
    default=defaults.excel,
    required=defaults.excel is None,
    help="Input Excel file path.",
  )
  copy_parser.add_argument(
    "--sheet",
    default=defaults.sheet,
    required=defaults.sheet is None,
    help="Worksheet name to read.",
  )
  copy_parser.add_argument(
    "--column",
    default=defaults.column,
    required=defaults.column is None,
    help="Column name that stores embedding file paths.",
  )
  copy_parser.add_argument(
    "--old-prefix",
    default=defaults.old_prefix,
    required=defaults.old_prefix is None,
    help="Old path prefix.",
  )
  copy_parser.add_argument(
    "--new-prefix",
    default=defaults.new_prefix,
    required=defaults.new_prefix is None,
    help="New path prefix.",
  )
  copy_parser.add_argument(
    "--target-root",
    default=defaults.target_root,
    required=defaults.target_root is None,
    help="Destination cache root directory.",
  )
  copy_parser.add_argument(
    "--log-file",
    default=defaults.log_file,
    required=defaults.log_file is None,
    help="Log file path for copy output and run summary.",
  )
  copy_parser.add_argument(
    "--verified-file-name",
    default=defaults.verified_file_name,
    required=defaults.verified_file_name is None,
    help="Verified JSONL basename under target root.",
  )
  copy_parser.add_argument(
    "--write-excel",
    default=None,
    help=(
      "Optional output Excel path. If provided, a copy with mapped EMBEDDING "
      "values (in the target sheet) is written."
    ),
  )
  copy_parser.add_argument(
    "--manifest",
    default=None,
    help="Optional manifest path to keep generated relative pkl list.",
  )
  copy_parser.add_argument(
    "--workers",
    type=int,
    default=defaults.workers,
    required=defaults.workers is None,
    help="Worker threads for Python copy flow.",
  )
  copy_parser.add_argument(
    "--verify-md5",
    action="store_true",
    help=(
      "Enable strict MD5 verification for existing cached files. Files copied "
      "in this run are always verified by size and MD5 before being marked done."
    ),
  )
  copy_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Preview copy operations only, do not transfer files.",
  )
  copy_parser.add_argument(
    "--status-only",
    action="store_true",
    help=(
      "Only show current cache status (cached pkl count + latest copy summary "
      "from log), without running copy."
    ),
  )
  copy_parser.set_defaults(handler=run_copy)

  link_parser = subparsers.add_parser(
    "link-valid",
    help="Create symlink tree for currently verified valid pkl files.",
    description=(
      "Create a symlink tree for currently verified shzs embedding pkl files, "
      "preserving the original relative directory structure."
    ),
  )
  link_parser.add_argument(
    "--cache-root",
    default=defaults.target_root,
    required=defaults.target_root is None,
    help="Root directory that stores cached pkl files.",
  )
  link_parser.add_argument(
    "--valid-root",
    default=defaults.valid_root,
    required=defaults.valid_root is None,
    help="Destination root for valid-file symlinks.",
  )
  link_parser.add_argument(
    "--verified-file",
    default=None,
    help=(
      "Verified JSONL path. Defaults to "
      "<cache-root>/.cache_shzs_embedding_verified.jsonl."
    ),
  )
  link_parser.add_argument(
    "--verified-file-name",
    default=defaults.verified_file_name,
    required=defaults.verified_file_name is None,
    help="Verified JSONL basename under cache root when --verified-file is omitted.",
  )
  link_parser.add_argument(
    "--replace",
    action="store_true",
    help="Replace wrong existing symlinks. Regular files/directories are never overwritten.",
  )
  link_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be linked without creating or replacing symlinks.",
  )
  link_parser.set_defaults(handler=run_link_valid)

  status_parser = subparsers.add_parser(
    "status",
    help="Show cache directory status and latest copy summary from log file.",
  )
  status_parser.add_argument(
    "--target-root",
    default=defaults.target_root,
    required=defaults.target_root is None,
    help="Cache root directory.",
  )
  status_parser.add_argument(
    "--log-file",
    default=defaults.log_file,
    required=defaults.log_file is None,
    help="Log file path generated by copy command.",
  )
  status_parser.set_defaults(handler=run_status)

  return parser


def main(
  argv: Optional[Sequence[str]] = None,
  *,
  defaults: Optional[CliDefaults] = None,
) -> int:
  parser = build_parser(defaults=defaults)
  args = parser.parse_args(argv)

  if getattr(args, "command", None) is None:
    parser.print_help()
    return 0

  return int(args.handler(args))


if __name__ == "__main__":
  raise SystemExit(main())

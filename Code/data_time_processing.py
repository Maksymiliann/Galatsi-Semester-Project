#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-align two videos using per-frame timestamps contained in companion text files.

Outputs:
  - CSV with matched frames (idx1, idx2, t1, t2, dt_ms)
  - two trimmed, time-aligned video files

Usage:
  python align_videos_by_time.py \
      --video1 path/to/videoA.mp4 --meta1 path/to/videoA_meta.txt \
      --video2 path/to/videoB.mp4 --meta2 path/to/videoB_meta.txt \
      --out_csv aligned_pairs.csv \
      --tolerance_ms 25 \
      --write_trimmed \
      --out_dir ./aligned_outputs

Place this file in: code/data_time_processing/ (or wherever you prefer)
"""

import argparse
import os
import re
from datetime import datetime, timedelta

import cv2
import numpy as np


# --------- Parsing ---------

# Takes input 2020-06-30 13:36:16,718,274 and return {
#   "date": "2020-06-30",
#   "hms": "13:36:16",
#   "ms": "718",
#   "us": "274"
# }

TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<hms>\d{2}:\d{2}:\d{2}),(?P<ms>\d{3}),(?P<us>\d{3})"
)
# If the files sometimes have just ",mmm" (no extra ",uuu"), this regex will handle it too:
TIMESTAMP_RE_FALLBACK = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<hms>\d{2}:\d{2}:\d{2}),(?P<ms>\d{3})(?!,)"
)


def parse_timestamps(meta_path):
    """Return list of (index, datetime) parsed from a metadata file."""
    ts = []
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        # Check for the time format
        for line in f:
            m = TIMESTAMP_RE.search(line)
            if m:
                d = m.group("date")
                hms = m.group("hms")
                ms = int(m.group("ms"))
                us = int(m.group("us"))
                dt = datetime.strptime(f"{d} {hms}", "%Y-%m-%d %H:%M:%S").replace(
                    microsecond=ms * 1000 + us
                )
                ts.append(dt)
                continue
            # fallback format (no extra microsecond chunk)
            m2 = TIMESTAMP_RE_FALLBACK.search(line)
            if m2:
                d = m2.group("date")
                hms = m2.group("hms")
                ms = int(m2.group("ms"))
                dt = datetime.strptime(f"{d} {hms}", "%Y-%m-%d %H:%M:%S").replace(
                    microsecond=ms * 1000
                )
                ts.append(dt)
    # Ensure strictly increasing and keep index
    ts_sorted = sorted(enumerate(ts), key=lambda x: x[1])
    return ts_sorted  # list of (frame_idx_in_meta_order, datetime)


# --------- Overlap Timestamp ---------

def compute_overlap_and_pairs(ts1, ts2, tolerance_ms=25):
    """
    ts1, ts2: lists of (idx, datetime)
    tolerance_ms: max time delta for pairing a frame from video1 to nearest in video2

    Returns:
      overlap_start, overlap_end (datetime)
      pairs: list of (idx1, idx2, t1, t2, dt_ms)
    """
    if not ts1 or not ts2:
        return None, None, []

    # Look at the start and end of each video
    start1, end1 = ts1[0][1], ts1[-1][1]
    start2, end2 = ts2[0][1], ts2[-1][1]

    # Determine the overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_end <= overlap_start:
        return None, None, []

    # two-pointer nearest-neighbor match within overlap window
    i, j = 0, 0
    pairs = []
    tol = timedelta(milliseconds=tolerance_ms)

    # advance pointers into overlap window
    while i < len(ts1) and ts1[i][1] < overlap_start:
        i += 1
    while j < len(ts2) and ts2[j][1] < overlap_start:
        j += 1

    while i < len(ts1) and j < len(ts2):
        t1 = ts1[i][1]
        t2 = ts2[j][1]
        if t1 > overlap_end or t2 > overlap_end:
            break

        # Find better of (j) or (j+1) for current t1
        best_j = j
        best_dt = abs(t1 - t2)
        if j + 1 < len(ts2):
            dt_next = abs(t1 - ts2[j + 1][1])
            if dt_next <= best_dt:
                best_j = j + 1
                best_dt = dt_next

        if best_dt <= tol:
            pairs.append(
                (ts1[i][0], ts2[best_j][0], t1, ts2[best_j][1], best_dt.total_seconds() * 1000.0)
            )
            i += 1
            j = best_j + 1  # move forward on ts2
        else:
            # advance the earlier timestamp
            if t1 < ts2[best_j][1]:
                i += 1
            else:
                j = best_j + 1

    return overlap_start, overlap_end, pairs


def write_pairs_srt(pairs, out_srt, fps=25):
    """
    Write aligned pairs in SRT subtitle format.
    Each block = index, time window, alignment info.
    """
    def format_time(ms):
        td = timedelta(milliseconds=ms)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    with open(out_srt, "w", encoding="utf-8") as f:
        for k, (i1, i2, t1, t2, dt) in enumerate(pairs, start=1):
            # Estimate duration (frame-to-frame time ~ 1/fps)
            start_ms = (k - 1) * (1000 / fps)
            end_ms   = k * (1000 / fps)
            f.write(f"{k}\n")
            f.write(f"{format_time(start_ms)} --> {format_time(end_ms)}\n")
            f.write(f"Video1 frame={i1} @ {t1}\n")
            f.write(f"Video2 frame={i2} @ {t2}\n")
            f.write(f"Δt = {dt:.3f} ms\n\n")



# --------- Trim videos to overlap (optional) ---------

def trim_video_by_indices(in_path, out_path, keep_indices, fps_override=None):
    """
    Save a trimmed video containing only frames at indices 'keep_indices'

    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps_override or fps_in

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    needed = set(keep_indices)
    idx = 0
    wrote = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            out.write(frame)
            wrote += 1
        idx += 1

    cap.release()
    out.release()
    return wrote


# --------- CLI ---------

def main():
    # paths for the videos and metadata
    video1 = "Dataset/Galatsi_Data_Semester_Project_archive/DJI_0808.MOV"
    meta1  = "Dataset/Galatsi_Data_Semester_Project_archive/DJI_0808.SRT"
    video2 = "Dataset/Galatsi_Data_Semester_Project_archive/DJI_0004.MP4"
    meta2  = "Dataset/Galatsi_Data_Semester_Project_archive/DJI_0004.SRT"

    out_srt = "Dataset/Galatsi_Data_Semester_Project_processing/aligned_pairs.SRT"
    out_dir = "Dataset/Galatsi_Data_Semester_Project_processing/aligned_outputs"
    tolerance_ms = 25
    write_trimmed = True  # set to False if you don’t want to output trimmed videos

    # === keep the rest of the pipeline the same ===
    ts1 = parse_timestamps(meta1)
    ts2 = parse_timestamps(meta2)

    overlap_start, overlap_end, pairs = compute_overlap_and_pairs(ts1, ts2, tolerance_ms)
    if not pairs:
        raise SystemExit("No overlapping pairs within tolerance. Try increasing tolerance_ms.")

    print(f"Overlap: {overlap_start}  →  {overlap_end}  ({len(pairs)} matched frames)")
    write_pairs_srt(pairs, out_srt)
    print(f"Wrote mapping SRT: {out_srt}")

    if write_trimmed:
        keep1 = [p[0] for p in pairs]
        keep2 = [p[1] for p in pairs]

        out1 = f"{out_dir}/DJI_0808_aligned.mp4"
        out2 = f"{out_dir}/DJI_0004_aligned.mp4"

        n1 = trim_video_by_indices(video1, out1, keep1)
        n2 = trim_video_by_indices(video2, out2, keep2)

        print(f"Wrote aligned trims: {out1} ({n1} frames), {out2} ({n2} frames)")



if __name__ == "__main__":
    main()

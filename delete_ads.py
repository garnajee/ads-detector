#!/usr/bin/env python3

import subprocess
import json
from datetime import datetime, timedelta

def to_seconds(timestamp):
    t = datetime.strptime(timestamp, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def to_timestamp(seconds):
    return str(timedelta(seconds=seconds))

def get_video_duration(input_file):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return int(float(result.stdout.strip()))

def adjust_remove_ranges(ranges):
    adjusted = []
    for i, (start, end) in enumerate(ranges):
        start_sec = to_seconds(start)
        end_sec = to_seconds(end)

        if i == 0:
            # Retirer 5 sec à la fin du premier segment
            adjusted.append((start, to_timestamp(max(0, end_sec - 5))))
        else:
            # Ajouter 5 sec au début des autres segments
            adjusted.append((to_timestamp(start_sec + 5), end))
    return adjusted

def build_keep_ranges(duration, remove_ranges):
    keep_ranges = []
    current_start = 0

    for start_str, end_str in remove_ranges:
        start = to_seconds(start_str)
        end = to_seconds(end_str)

        if current_start < start:
            keep_ranges.append((to_timestamp(current_start), to_timestamp(start)))

        current_start = end

    if current_start < duration:
        keep_ranges.append((to_timestamp(current_start), to_timestamp(duration)))

    return keep_ranges

def run_mkvmerge(input_file, output_file, keep_ranges):
    parts_str = ",".join(f"+{start}-{end}" if i else f"{start}-{end}" for i, (start, end) in enumerate(keep_ranges))

    cmd = [
        "mkvmerge",
        "-o", output_file,
        "--split", f"parts:{parts_str}",
        input_file
    ]

    subprocess.run(cmd, check=True)
    print("✅ Montage terminé avec succès :", output_file)

def main(input_file, output_file, original_remove_ranges):
    if input_file and output_file and original_remove_ranges:
        print("[delete-ads.py] - need to add input/output/ranges.")

    adjusted_remove_ranges = adjust_remove_ranges(original_remove_ranges)
    duration = get_video_duration(input_file)
    keep_ranges = build_keep_ranges(duration, adjusted_remove_ranges)
    run_mkvmerge(input_file, output_file, keep_ranges)

if __name__ == "__main__":
    input_file = "video.mkv"
    output_file = "output-no-ads.mkv"

    original_remove_ranges = [
        ("00:00:00", "00:03:51"),
        ("00:18:42", "00:23:46"),
        ("00:46:50", "00:50:05"),
        ("01:17:16", "01:20:26"),
        ("01:48:50", "01:56:57"),
        ("02:14:20", "02:21:38"),
        ("02:43:09", "02:49:28"),
        ("02:52:29", "02:56:51"),
    ]

    main(input_file, output_file, original_remove_ranges)

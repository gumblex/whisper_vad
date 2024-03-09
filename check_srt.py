#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import subprocess

re_timestamp = re.compile(r'^(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)')


def get_media_duration(filename):
    proc = subprocess.run((
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', '--', filename
    ), capture_output=True)
    return float(proc.stdout.strip().decode())


def get_srt_duration(filename):
    max_ts = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for ln in f:
            match = re_timestamp.match(ln)
            if not match:
                continue
            d = [int(x) for x in match.groups()[4:]]
            ts = d[0] * 3600 + d[1] * 60 + d[2] + d[3] * 0.001
            if ts > max_ts:
                max_ts = ts
    return max_ts


def main(path):
    for root, dirs, files in os.walk(path):
        srt_files = set()
        media_files = {}
        for filename in files:
            basename, fext = os.path.splitext(filename)
            if fext == '.srt':
                srt_files.add(basename)
            elif fext in ('.mp3', '.m4a', '.flac', '.mp4', '.mkv', '.mka', '.avi'):
                media_files[basename] = filename
        for basename in sorted(srt_files.intersection(media_files.keys())):
            srt_path = os.path.join(root, basename + '.srt')
            srt_length = get_srt_duration(srt_path)
            media_path = os.path.join(root, media_files[basename])
            media_length = get_media_duration(media_path)
            if ((srt_length < media_length * 0.9 and srt_length < media_length - 10)
                or srt_length > media_length + 60
                or os.stat(srt_path).st_mtime < 1673882348):
                #print(media_path, media_length, srt_length)
                print(media_path)
        for basename in sorted(media_files.keys()):
            if basename in srt_files:
                continue
            print(os.path.join(root, media_files[basename]))


if __name__ == '__main__':
    main(sys.argv[1])

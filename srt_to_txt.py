#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import subprocess

re_num = re.compile(r'^(\d+)$')
re_timestamp = re.compile(r'^(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)')

PARAGRAPH_INTERVAL = 1


def srt_to_txt(fp):
    state = 0
    current_ts = 0
    for ln in fp:
        ln = ln.strip()
        if state == 0:
            if re_num.match(ln):
                state = 1
        elif state == 1:
            match = re_timestamp.match(ln)
            if match:
                d = [int(x) for x in match.groups()]
                ts1 = d[0] * 3600 + d[1] * 60 + d[2] + d[3] * 0.001
                ts2 = d[4] * 3600 + d[5] * 60 + d[6] + d[7] * 0.001
                if current_ts > 0 and (
                    ts1 - current_ts > PARAGRAPH_INTERVAL or ts1 < current_ts
                ):
                    yield ''
                current_ts = ts2
                state = 2
        elif state == 2:
            if not ln:
                state = 0
                continue
            yield ln.strip()


if __name__ == '__main__':
    for line in srt_to_txt(sys.stdin):
        print(line)

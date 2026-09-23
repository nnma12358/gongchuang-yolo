#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多路相机接口冒烟测试（离线可用，配合 scripts/dev_mock_cameras.py）

检查三件事：
  1. 每一路都能取到单帧（JPEG 魔数正确）
  2. 原生 MJPEG 的源（main/marked/cam2）能持续吐帧
  3. 没有原生 MJPEG 的源（depth）由网关按帧率合成 multipart，同样能吐帧

用法：
    python3 scripts/smoke_cameras.py --base http://127.0.0.1:8099 --seconds 4
"""
import argparse
import sys
import time

import requests


def count_frames(url, seconds, chunk=8192):
    """读一段 MJPEG，统计收到的 JPEG 帧数（按 SOI 魔数计数）"""
    n = 0
    t0 = time.time()
    got = 0
    with requests.get(url, stream=True, timeout=(3.05, 10)) as r:
        if r.status_code != 200:
            return -1, r.status_code
        buf = b''
        for data in r.iter_content(chunk_size=chunk):
            if not data:
                continue
            buf += data
            n += buf.count(b'\xff\xd8\xff')
            buf = buf[-8:]
            got += len(data)
            if time.time() - t0 >= seconds:
                break
    return n, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8099')
    ap.add_argument('--seconds', type=float, default=4.0)
    args = ap.parse_args()

    r = requests.get(args.base + '/api/cameras', timeout=8)
    r.raise_for_status()
    cams = r.json()['cameras']
    print('发现 {0} 路相机'.format(len(cams)))

    fails = 0
    for c in cams:
        cid = c['id']
        # 1) 单帧
        snap = requests.get(args.base + c['snapshot_url'], timeout=5)
        ok_jpeg = snap.status_code == 200 and snap.content[:2] == b'\xff\xd8'
        print('  [{0}] 单帧 {1}  {2} 字节{3}'.format(
            cid, 'OK' if ok_jpeg else '失败', len(snap.content),
            '' if ok_jpeg else ' HTTP {0}'.format(snap.status_code)))
        if not ok_jpeg:
            fails += 1
        # 2) 流
        n, code = count_frames(args.base + c['stream_url'], args.seconds)
        ok_stream = n >= 2
        print('  [{0}] 流   {1}  {2:.0f}s 收到 {3} 帧 (HTTP {4})'.format(
            cid, 'OK' if ok_stream else '失败', args.seconds, n, code))
        if not ok_stream:
            fails += 1

    print('\n结论：{0}'.format('全部通过' if fails == 0 else '{0} 项失败'.format(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

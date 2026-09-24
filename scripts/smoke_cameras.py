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
    """读一段 MJPEG，统计收到的 JPEG 帧数（按 SOI 魔数计数）

    上游静默/断开不抛异常，返回已收到的帧数 —— 这样一路失败不影响其它路的报告。
    """
    n = 0
    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=(3.05, max(6.0, seconds))) as r:
            if r.status_code != 200:
                return -1, r.status_code
            buf = b''
            for data in r.iter_content(chunk_size=chunk):
                if not data:
                    continue
                buf += data
                n += buf.count(b'\xff\xd8\xff')
                buf = buf[-8:]
                if time.time() - t0 >= seconds:
                    break
    except Exception as e:
        return n, 'ERR {0}'.format(str(e)[:40])
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
        try:
            snap = requests.get(args.base + c['snapshot_url'], timeout=6)
            ok_jpeg = snap.status_code == 200 and snap.content[:2] == b'\xff\xd8'
            detail = '' if ok_jpeg else ' HTTP {0}'.format(snap.status_code)
            size = len(snap.content)
        except Exception as e:
            ok_jpeg, detail, size = False, ' {0}'.format(str(e)[:40]), 0
        print('  [{0}] 单帧 {1}  {2} 字节{3}'.format(cid, 'OK' if ok_jpeg else '失败', size, detail))
        if not ok_jpeg:
            fails += 1
        # 2) 流
        n, code = count_frames(args.base + c['stream_url'], args.seconds)
        ok_stream = n >= 2
        print('  [{0}] 流   {1}  {2:.0f}s 收到 {3} 帧 ({4})'.format(
            cid, 'OK' if ok_stream else '失败', args.seconds, n, code))
        if not ok_stream:
            fails += 1

    print('\n结论：{0}'.format(
        '全部通过' if fails == 0 else '{0} 项失败（不可达的源正常失败，属预期）'.format(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

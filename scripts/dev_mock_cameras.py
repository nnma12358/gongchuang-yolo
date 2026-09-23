#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线联调用的模拟相机服务 —— 在 PC 上模拟 Jetson 端的上游画面源，
用于在没有设备的情况下验证「多路相机 / 前端双画面」这条链路。

它模拟的是**上游接口形状**（不是业务数据）：
    模拟 ROS 桥接：  /color.jpg  /color.mjpg  /depth_preview.jpg
    模拟视觉容器：    /frame.jpg  /stream.mjpg

用法：
    python3 scripts/dev_mock_cameras.py --port 8192 --mode bridge
    python3 scripts/dev_mock_cameras.py --port 8194 --mode vision
    # 然后让网关指向它们：
    BRIDGE_URL=http://127.0.0.1:8192 VISION_URL=http://127.0.0.1:8194 \
        python3 server/gateway.py         # 需要 PORT=8099 等环境变量

画面里带帧号与时间戳，便于一眼确认「流是活的、没有卡在某一帧」。
"""
import argparse
import io
import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse


def make_frame(port, seq, mode, w=640, h=480):
    """画一帧带帧号的测试图（不同 mode 用不同底色，便于区分是哪一路）"""
    import cv2
    base = {'bridge': (40, 60, 90), 'vision': (30, 80, 50), 'depth': (90, 60, 40)}.get(mode, (50, 50, 50))
    img = np.full((h, w, 3), base, np.uint8)
    # 移动方块：证明帧在更新
    x = int((seq * 7) % (w - 80))
    cv2.rectangle(img, (x, 60), (x + 80, 140), (60, 200, 255), -1)
    cv2.putText(img, '{0} #{1}'.format(mode.upper(), seq), (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, 'port {0}  {1}'.format(port, time.strftime('%H:%M:%S')), (16, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description="离线联调模拟相机")
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--mode', default='bridge', choices=['bridge', 'vision', 'depth'])
    ap.add_argument('--fps', type=float, default=10.0)
    args = ap.parse_args()

    import cv2
    import uvicorn

    app = FastAPI(docs_url=None, redoc_url=None)
    lock = threading.Lock()
    state = {'seq': 0, 'jpg': None}

    def render_loop():
        period = 1.0 / max(1.0, args.fps)
        while True:
            with lock:
                state['seq'] += 1
                seq = state['seq']
            img = make_frame(args.port, seq, args.mode)
            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with lock:
                    state['jpg'] = buf.tobytes()
            time.sleep(period)

    def snapshot():
        with lock:
            return state['jpg'], state['seq']

    def jpg_response():
        jpg, _ = snapshot()
        if jpg is None:
            return JSONResponse({'detail': '尚无帧'}, status_code=503)
        return Response(content=jpg, media_type='image/jpeg',
                        headers={'Cache-Control': 'no-store'})

    @app.get('/health')
    def health():
        _, seq = snapshot()
        return {'ok': True, 'mock': True, 'mode': args.mode, 'seq': seq}

    if args.mode == 'bridge':
        app.get('/color.jpg')(jpg_response)
        app.get('/depth_preview.jpg')(jpg_response)

        @app.get('/color.mjpg')
        def color_mjpg():
            boundary = b'mockframe'

            def gen():
                last = -1
                while True:
                    jpg, seq = snapshot()
                    if jpg is not None and seq != last:
                        last = seq
                        yield (b'--' + boundary + b'\r\nContent-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / 30.0)

            return StreamingResponse(gen(),
                                     media_type='multipart/x-mixed-replace; boundary={0}'.format(
                                         boundary.decode()))
    else:
        app.get('/frame.jpg')(jpg_response)

        @app.get('/stream.mjpg')
        def stream_mjpg():
            boundary = b'mockframe'

            def gen():
                last = -1
                while True:
                    jpg, seq = snapshot()
                    if jpg is not None and seq != last:
                        last = seq
                        yield (b'--' + boundary + b'\r\nContent-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / 30.0)

            return StreamingResponse(gen(),
                                     media_type='multipart/x-mixed-replace; boundary={0}'.format(
                                         boundary.decode()))

    threading.Thread(target=render_loop, daemon=True).start()
    time.sleep(0.5)
    print('模拟相机 {0} 启动于 :{1}'.format(args.mode, args.port), flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()

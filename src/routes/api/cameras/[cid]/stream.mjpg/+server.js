import { gatewayStream } from '$lib/server/proxy.js';

/**
 * 单路 MJPEG 流 —— 长连接，前端 <img src> 直接显示。
 * 必须走 gatewayStream（无超时），否则 6s 后会被掐断、画面卡死。
 */
export function GET({ params }) {
	return gatewayStream(`/api/cameras/${params.cid}/stream.mjpg`);
}

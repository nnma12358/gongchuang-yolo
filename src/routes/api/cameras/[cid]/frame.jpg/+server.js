import { gateway } from '$lib/server/proxy.js';

/** 单路相机当前帧（抓拍 / 轮询用） */
export function GET({ params }) {
	return gateway(`/api/cameras/${params.cid}/frame.jpg`);
}

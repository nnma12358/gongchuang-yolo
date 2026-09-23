import { gateway } from '$lib/server/proxy.js';

/** 相机清单（含真实可用状态）—— 前端据此渲染可选的画面源 */
export function GET({ url }) {
	return gateway('/api/cameras', { search: url.search });
}

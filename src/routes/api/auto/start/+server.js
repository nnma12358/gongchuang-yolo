import { gateway } from '$lib/server/proxy.js';

/** 自动分拣（循环运行在 Jetson 网关容器内，PC 端仅转发控制指令） */
export async function POST({ request }) {
	const body = await request.text();
	return gateway('/api/auto/start', { method: 'POST', body });
}

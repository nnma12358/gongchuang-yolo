import { json } from '@sveltejs/kit';
import { deployment, GATEWAY_URL } from '$lib/server/proxy.js';

/**
 * PC 调试容器自身健康检查 + 部署信息
 * 说明：PC 容器只提供前端页面与代理；分拣/识别/自动分拣均在 Jetson 端，
 * 因此这里还会探测真实服务是否可用，供前端显示“已连接 / 未部署”。
 */
export async function GET() {
	const info = deployment();
	let gateway = { reachable: false };
	if (GATEWAY_URL) {
		try {
			const r = await fetch(`${GATEWAY_URL}/health`, { signal: AbortSignal.timeout(2500) });
			const d = await r.json().catch(() => ({}));
			gateway = { reachable: r.ok, vision: d.vision || null, auto: d.auto || null };
		} catch (e) {
			gateway = { reachable: false, error: e.message };
		}
	}
	return json({
		ok: true,
		role: 'PC 前端调试容器（ui-shell）',
		...info,
		gateway_reachable: gateway.reachable,
		vision: gateway.vision || null,
		auto: gateway.auto || null
	});
}

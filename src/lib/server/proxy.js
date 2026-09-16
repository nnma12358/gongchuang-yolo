/**
 * PC 调试容器 —— 服务代理（唯一数据来源：真实部署的服务）
 * ------------------------------------------------------------------
 * PC 容器不含任何分拣逻辑与模拟数据：
 *   · GATEWAY_URL  指向 Jetson 网关容器（:80）—— 显示数据 / 任务 / 轮次 / 自动分拣
 *   · VISION_URL   指向 Jetson 视觉容器（:8100）—— 画面帧 / 标记
 * 未配置或不可达时，接口返回 503 + 结构化说明，前端据此显示“未部署/未连接”，
 * 绝不展示本地伪数据（以实际服务为准）。
 */
import { json } from '@sveltejs/kit';

export const GATEWAY_URL = (process.env.GATEWAY_URL || '').replace(/\/$/, '');
export const VISION_URL = (process.env.VISION_URL || '').replace(/\/$/, '');

export function deployment() {
	return {
		pc_role: 'ui-shell',
		gateway_url: GATEWAY_URL || null,
		vision_url: VISION_URL || null,
		deployed: Boolean(GATEWAY_URL)
	};
}

function unconfigured(target, env) {
	return json(
		{
			deployed: false,
			configured: false,
			target,
			detail:
				`未连接实际服务：${target} 未配置。PC 调试容器只提供前端页面与代理，` +
				`分拣/识别/自动分拣都在 Jetson 端运行 —— 请设置环境变量 ${env} 指向 Jetson 后重启容器。`,
			hint: `在 sort-web/.env 中填写 ${env}=http://<jetson-ip>:<port>，然后 docker compose up -d`
		},
		{ status: 503 }
	);
}

async function forward(base, path, { method = 'GET', body, search = '', timeoutMs = 6000 } = {}) {
	const url = `${base}${path}${search || ''}`;
	const ctl = AbortSignal.timeout(timeoutMs);
	const init = { method, signal: ctl, headers: {} };
	if (body && method !== 'GET' && method !== 'HEAD') {
		init.headers['Content-Type'] = 'application/json';
		init.body = typeof body === 'string' ? body : JSON.stringify(body);
	}
	const resp = await fetch(url, init);
	const type = resp.headers.get('content-type') || 'application/json';
	if (!/^(application\/json|text\/)/i.test(type)) {
		// 二进制（JPEG 等）按字节透传
		return new Response(resp.body, {
			status: resp.status,
			headers: { 'Content-Type': type, 'Cache-Control': 'no-store, max-age=0' }
		});
	}
	return new Response(await resp.text(), {
		status: resp.status,
		headers: { 'Content-Type': type, 'Cache-Control': 'no-store' }
	});
}

async function unreachable(target, base, err) {
	return json(
		{ deployed: Boolean(GATEWAY_URL), configured: true, target, detail: `${target} 不可达（${base}）：${err.message}` },
		{ status: 502 }
	);
}

/** 代理到 Jetson 网关容器（分拣显示数据 / 任务 / 轮次 / 自动分拣 / 图库） */
export async function gateway(path, opts = {}) {
	if (!GATEWAY_URL) return unconfigured('Jetson 网关容器', 'GATEWAY_URL');
	try {
		return await forward(GATEWAY_URL, path, opts);
	} catch (e) {
		return unreachable('Jetson 网关容器', GATEWAY_URL, e);
	}
}

/** 代理到 Jetson 视觉容器（画面帧 / 标记 / 健康状态） */
export async function vision(path, opts = {}) {
	if (!VISION_URL) {
		// 视觉容器地址可由网关透传获取（网关 /api/status 内含 vision_url）
		if (GATEWAY_URL) {
			try {
				const r = await fetch(`${GATEWAY_URL}/api/status`, { signal: AbortSignal.timeout(4000) });
				const d = await r.json();
				if (d && d.vision_url) {
					return await forward(String(d.vision_url).replace(/\/$/, ''), path, opts);
				}
			} catch (e) {
				return unreachable('Jetson 网关容器', GATEWAY_URL, e);
			}
		}
		return unconfigured('Jetson 视觉容器', 'VISION_URL');
	}
	try {
		return await forward(VISION_URL, path, opts);
	} catch (e) {
		return unreachable('Jetson 视觉容器', VISION_URL, e);
	}
}

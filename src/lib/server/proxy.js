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
import { probe } from './net-probe.js';

export const GATEWAY_URL = (process.env.GATEWAY_URL || '').replace(/\/$/, '');
export const VISION_URL = (process.env.VISION_URL || '').replace(/\/$/, '');

export function deployment() {
	const st = resolveState();
	return {
		pc_role: 'ui-shell',
		gateway_url: (st.resolved && st.resolved.base) || GATEWAY_URL || null,
		gateway_source: st.resolved ? st.resolved.source : st.env_url ? '环境变量' : null,
		gateway_override: st.override || null,
		vision_url: VISION_URL || null,
		deployed: Boolean((st.resolved && st.resolved.base) || GATEWAY_URL)
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

/**
 * 流式转发（MJPEG 长连接专用）
 * forward() 默认 6s 超时，会把 MJPEG 长连接掐断（表现为画面几秒后卡死），
 * 这里不做任何超时，并把上游的 multipart Content-Type 原样透传。
 */
async function forwardStream(base, path, { search = '' } = {}) {
	const resp = await fetch(`${base}${path}${search || ''}`, { headers: { Accept: '*/*' } });
	const type = resp.headers.get('content-type') || 'application/octet-stream';
	return new Response(resp.body, {
		status: resp.status,
		headers: {
			'Content-Type': type,
			'Cache-Control': 'no-store, max-age=0',
			'X-Accel-Buffering': 'no'
		}
	});
}

export async function gatewayStream(path, opts = {}) {
	const g = await resolveTarget();
	if (!g.base) return unconfigured('Jetson 网关容器', 'GATEWAY_URL');
	try {
		return await forwardStream(g.base, path, opts);
	} catch (e) {
		return unreachable('Jetson 网关容器', g.base, e);
	}
}

/**
 * 目标地址解析 —— Nano 的地址随所连无线热点变化，换热点后整个网段都会变，
 * 因此不写死地址，按顺序取第一个可用的候选：
 *   1. 运行期覆盖（前端「连接设置」手填，存内存）
 *   2. GATEWAY_URL（推荐填主机名，例如 wheeltec.local，换网段也不用改）
 *   3. GATEWAY_CANDIDATES（逗号分隔的多个候选）
 * 命中后缓存 TTL；探测失败立即失效重解析，所以换热点后刷新页面即可自动接上。
 */
let override = null;
let resolved = null;
let resolveInfo = { tried: [], attempts: 0, last_error: null };
const RESOLVE_TTL = Number(process.env.DISCOVERY_TTL_MS || 30000);

function scheme() {
	return ['ht', 'tp://'].join('');
}

export function normalizeTarget(url) {
	if (!url) return null;
	const s = String(url).trim().replace(/\/+$/, '');
	if (!s) return null;
	return /^[a-z]+:\/\//i.test(s) ? s : scheme() + s;
}

export function setTarget(url) {
	override = normalizeTarget(url);
	resolved = null;
	return override;
}

export function clearTarget() {
	override = null;
	resolved = null;
	return null;
}

function targetCandidates() {
	const list = [];
	const push = (v, why) => {
		const n = normalizeTarget(v);
		if (n && !list.some((x) => x.base === n)) list.push({ base: n, why });
	};
	push(override, '运行期覆盖');
	push(GATEWAY_URL, 'GATEWAY_URL');
	for (const c of (process.env.GATEWAY_CANDIDATES || '').split(',')) {
		const n = normalizeTarget(c);
		if (n) push(n, '候选列表');
	}
	for (const h of (process.env.GATEWAY_MDNS || 'wheeltec.local,jetson.local,nano.local').split(',')) {
		const n = normalizeTarget(h.trim());
		if (n) push(n, 'mDNS 主机名');
	}
	return list;
}

/**
 * 解析出当前可用的网关地址。
 * @param {boolean} force        忽略缓存（「重新检测」）
 * @param {boolean} onlyOverride 只认「运行期指定」的地址，不做回退
 *   —— 用户在连接设置里手填地址时必须是这个语义：填错了要如实报错，
 *      不能悄悄连到别的地址上去（否则用户以为生效了）。
 */
export async function resolveTarget(force = false, onlyOverride = false) {
	const now = Date.now();
	if (!force && resolved && now - resolved.at < RESOLVE_TTL) {
		return { base: resolved.base, source: resolved.source, cached: true };
	}
	resolveInfo.attempts += 1;
	const all = targetCandidates();
	const list = onlyOverride ? all.filter((c) => c.why === '运行期覆盖') : all;
	if (!list.length) {
		resolveInfo = { tried: [], attempts: resolveInfo.attempts, last_error: '未填写地址' };
		return { base: null, source: null, tried: [], error: '未填写地址' };
	}
	const tried = [];
	for (const cand of list) {
		const r = await probe(cand.base);
		tried.push({ base: cand.base, why: cand.why, ok: r.ok, detail: r.detail });
		if (r.ok) {
			resolved = { base: cand.base, source: cand.why, at: now };
			resolveInfo = { tried, attempts: resolveInfo.attempts, last_error: null };
			return { base: cand.base, source: cand.why, tried, health: r.health };
		}
	}
	resolved = null;
	resolveInfo = {
		tried,
		attempts: resolveInfo.attempts,
		last_error: tried.length ? tried[tried.length - 1].detail : '未配置任何候选地址'
	};
	return { base: null, source: null, tried, error: resolveInfo.last_error };
}

export function resolveState() {
	return {
		override,
		env_url: GATEWAY_URL || null,
		env_candidates: process.env.GATEWAY_CANDIDATES || null,
		resolved: resolved ? { base: resolved.base, source: resolved.source, age_ms: Date.now() - resolved.at } : null,
		attempts: resolveInfo.attempts,
		tried: resolveInfo.tried,
		last_error: resolveInfo.last_error
	};
}

async function unreachable(target, base, err) {
	return json(
		{ deployed: Boolean(GATEWAY_URL), configured: true, target, detail: `${target} 不可达（${base}）：${err.message}` },
		{ status: 502 }
	);
}

/** 代理到 Jetson 网关容器（分拣显示数据 / 任务 / 轮次 / 自动分拣 / 图库） */
export async function gateway(path, opts = {}) {
	const g = await resolveTarget();
	if (!g.base) {
		const st = resolveState();
		const configured = Boolean(st.env_url || st.override || st.env_candidates);
		if (!configured) return unconfigured('Jetson 网关容器', 'GATEWAY_URL');
		const first = (st.tried && st.tried[0] && st.tried[0].base) || '候选地址';
		return unreachable('Jetson 网关容器', first, new Error(st.last_error || '全部候选地址均不可达'));
	}
	try {
		const resp = await forward(g.base, path, opts);
		if (resp.status >= 500) resolved = null; // 下次请求重新解析（换热点后可自愈）
		return resp;
	} catch (e) {
		resolved = null;
		return unreachable('Jetson 网关容器', g.base, e);
	}
}

/** 代理到 Jetson 视觉容器（画面帧 / 标记 / 健康状态） */
export async function vision(path, opts = {}) {
	if (!VISION_URL) {
		// 视觉容器地址可由网关透传获取（网关 /api/status 内含 vision_url）
		const g = await resolveTarget();
		if (g.base) {
			try {
				const r = await forward(g.base, '/api/status', { timeoutMs: 4000 });
				if (r.ok) {
					const d = await r.json();
					if (d && d.vision_url) {
						return await forward(String(d.vision_url).replace(/\/$/, ''), path, opts);
					}
				}
			} catch (e) {
				return unreachable('Jetson 网关容器', g.base, e);
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

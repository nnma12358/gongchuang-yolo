/**
 * 网关地址探测（最小实现）—— 供 PC 端 ui-shell 解析 Jetson 网关地址。
 * 现场 Nano 的地址会随所连无线热点变化，因此不在代码里写死，改为运行期解析。
 */
export async function probe(base, ms = 1200) {
	try {
		const r = await fetch(base + '/health', { signal: AbortSignal.timeout(ms) });
		if (!r.ok) return { ok: false, detail: 'HTTP ' + r.status };
		const d = await r.json().catch(() => null);
		if (!d || d.service !== 'sort-gateway') return { ok: false, detail: '非分拣网关' };
		return { ok: true, health: d };
	} catch (e) {
		return { ok: false, detail: e && e.message ? e.message : String(e) };
	}
}

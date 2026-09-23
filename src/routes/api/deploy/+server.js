import { json } from '@sveltejs/kit';
import { clearTarget, resolveState, resolveTarget, setTarget } from '$lib/server/proxy.js';

/**
 * 连接设置 —— 解决「Nano 的地址随所连无线热点变化」：
 *   GET  ?refresh=1   强制重新解析（「重新检测」按钮）
 *   POST {url}        运行期指定网关地址（不必改 .env、不必重启容器）
 *   POST {clear:true} 清除运行期覆盖，回到环境变量配置
 */
export async function GET({ url }) {
	const force = url.searchParams.get('refresh') === '1';
	const r = await resolveTarget(force);
	return json({ ...resolveState(), ok: Boolean(r.base), base: r.base, source: r.source });
}

export async function POST({ request }) {
	const body = await request.json().catch(() => ({}));
	if (body && body.clear) {
		// 清除运行期覆盖 → 回到环境变量/候选列表，并整体重测
		clearTarget();
		const r = await resolveTarget(true);
		return json({
			ok: Boolean(r.base),
			action: 'clear',
			...resolveState(),
			base: r.base,
			source: r.source,
			hint: r.base ? `已自动找到服务：${r.base}` : `未找到服务：${r.error || '全部候选不可达'}`
		});
	}
	const want = body.url || body.target || '';
	if (!want) {
		const r = await resolveTarget(true);
		return json({ ok: Boolean(r.base), action: 'refresh', ...resolveState(), base: r.base, source: r.source });
	}
	// 显式指定地址：只认这一个，失败就如实报错（不回退到别的地址，避免"以为生效了"）
	setTarget(want);
	const r = await resolveTarget(true, true);
	return json({
		ok: Boolean(r.base),
		action: 'set',
		...resolveState(),
		base: r.base,
		source: r.source,
		hint: r.base
			? `已连接 ${r.base}`
			: `该地址不可用：${r.error || '未响应'}（需能返回 /health 且带 service=sort-gateway 标记）`
	});
}

import { vision } from '$lib/server/proxy.js';

/** 视觉容器当前帧（含标记）—— PC 显示屏显示 Jetson 端真实画面 */
export function GET({ url }) {
	const draw = url.searchParams.get('draw') === '0' ? 0 : 1;
	return vision(`/frame.jpg?draw=${draw}`);
}

import { vision } from '$lib/server/proxy.js';

/** 最近一帧标记（透传 Jetson 视觉容器） */
export function GET() {
	return vision('/marks/latest');
}

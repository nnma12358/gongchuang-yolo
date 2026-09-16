import { vision } from '$lib/server/proxy.js';

/** 视觉容器健康状态 */
export function GET() {
	return vision('/health');
}

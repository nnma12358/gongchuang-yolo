import { gateway } from '$lib/server/proxy.js';

export function GET() {
	return gateway('/api/auto/status');
}

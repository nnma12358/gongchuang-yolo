import { gateway } from '$lib/server/proxy.js';

export async function POST() {
	return gateway('/api/auto/stop', { method: 'POST', body: {} });
}

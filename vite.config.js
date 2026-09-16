import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// 说明：/api/* 由 SvelteKit 服务端路由（src/routes/api/**）直接提供，
//       dev 模式与生产模式行为一致，无需再代理到外部网关。
export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: true,
		port: 5173,
		strictPort: false
	}
});

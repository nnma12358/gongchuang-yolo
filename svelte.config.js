import adapterNode from '@sveltejs/adapter-node';
import adapterStatic from '@sveltejs/adapter-static';

/**
 * 双构建目标（同一套 Svelte 5 源码）：
 *   ADAPTER=node   （默认）→ 产物 build/ 为 Node 服务端应用：PC 端 node22 容器调试 /api 与页面
 *   ADAPTER=static        → 产物 build/ 为静态 SPA：交给 Jetson Nano 网关容器（Python）托管
 */
const useStatic = process.env.ADAPTER === 'static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: useStatic
			? adapterStatic({ fallback: 'index.html' })
			: adapterNode({ out: 'build' }),
		alias: {
			$lib: './src/lib'
		}
	}
};

export default config;

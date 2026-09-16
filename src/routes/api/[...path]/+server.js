import { gateway } from '$lib/server/proxy.js';

/**
 * 兜底代理：/api/** 全部转发到 Jetson 网关容器（真实服务）
 * PC 调试容器不含任何分拣逻辑，未连接时返回 503 + 说明（前端显示“未部署”）
 */
const GET = ({ params, url }) => gateway(`/api/${params.path}`, { search: url.search });
const POST = ({ params, url, request }) => gateway(`/api/${params.path}`, { method: 'POST', body: request });
const PUT = ({ params, url, request }) => gateway(`/api/${params.path}`, { method: 'PUT', body: request });
const DELETE = ({ params, url }) => gateway(`/api/${params.path}`, { method: 'DELETE' });

export { GET, POST, PUT, DELETE };

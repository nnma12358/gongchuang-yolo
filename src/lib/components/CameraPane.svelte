<script>
	/**
	 * 单路相机窗格 —— 一个窗格里包含：源选择 / 实时画面 / 状态 / 抓拍 / 全屏。
	 * 页面上放两个（或更多）即可实现「同时显示多路摄像头画面」。
	 *
	 * 画面来源分三类：
	 *   gateway  经由网关的多路相机（main/marked/depth/cam2…），走 MJPEG <img>
	 *   local    浏览器本机摄像头（getUserMedia，用于扫码/临时拍摄）
	 *   custom   手填的任意 MJPEG 地址
	 */
	let {
		title = '画面',
		cameras = [],
		value = $bindable(''),
		oncapture = null,
		showDetect = false,
		detecting = false,
		ondetect = null,
		compact = false
	} = $props();

	let tick = $state(0);
	let online = $state(false);
	let errorText = $state('');
	let videoEl = $state(undefined);
	let wrapEl = $state(undefined);
	let localStream = null;
	let retryTimer = null;

	const isLocal = $derived(value === 'local');
	const isCustom = $derived(value === 'custom');
	const cam = $derived(cameras.find((c) => c.id === value) || null);
	const streamUrl = $derived(!isLocal && !isCustom && cam ? `${cam.stream_url}?t=${tick}` : '');

	/** 状态行：优先用网关探测到的真实信息，其次用画面自身的加载结果 */
	const status = $derived.by(() => {
		if (isLocal) return { text: localStream ? '本机摄像头已开启' : '未开启', ok: Boolean(localStream) };
		if (isCustom) return { text: customUrl ? '自定义流' : '未填写地址', ok: Boolean(customUrl) };
		if (!cam) return { text: '未选择画面源', ok: false };
		if (!cam.ready) return { text: cam.error ? `不可达：${cam.error}` : '不可达', ok: false };
		const size = cam.size && cam.size[0] ? `${cam.size[0]}×${cam.size[1]}` : '';
		return { text: ['在线', size, cam.latency_ms ? `${cam.latency_ms}ms` : ''].filter(Boolean).join(' · '), ok: true };
	});

	let customUrl = $state('');

	export function reload() {
		online = false;
		errorText = '';
		tick += 1;
	}

	export function fullscreen() {
		if (!wrapEl) return;
		if (document.fullscreenElement) document.exitFullscreen?.();
		else wrapEl.requestFullscreen?.();
	}

	/** 取当前画面的一帧（JPEG Blob）——远程相机直接向网关要单帧，本机摄像头走 canvas */
	export async function grabFrame() {
		if (isLocal) {
			if (!videoEl?.videoWidth) return null;
			const cv = document.createElement('canvas');
			cv.width = videoEl.videoWidth;
			cv.height = videoEl.videoHeight;
			cv.getContext('2d').drawImage(videoEl, 0, 0);
			const blob = await new Promise((r) => cv.toBlob(r, 'image/jpeg', 0.9));
			return { blob, url: URL.createObjectURL(blob) };
		}
		const url = isCustom ? customUrl : cam ? cam.snapshot_url : '';
		if (!url) return null;
		const resp = await fetch(url + (url.includes('?') ? '&' : '?') + 't=' + Date.now());
		if (!resp.ok) throw new Error(`取帧失败 HTTP ${resp.status}`);
		const blob = await resp.blob();
		return { blob, url: URL.createObjectURL(blob) };
	}

	async function openLocal() {
		if (localStream) return;
		try {
			localStream = await navigator.mediaDevices.getUserMedia({
				video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
				audio: false
			});
			if (videoEl) videoEl.srcObject = localStream;
		} catch (e) {
			errorText = '无法访问本机摄像头：' + (e.message || e);
		}
	}

	function closeLocal() {
		if (localStream) localStream.getTracks().forEach((t) => t.stop());
		localStream = null;
		online = false;
	}

	function onImgLoad() {
		online = true;
		errorText = '';
		clearTimeout(retryTimer);
	}
	function onImgError() {
		online = false;
		errorText = '画面中断，正在自动重连…';
		clearTimeout(retryTimer);
		retryTimer = setTimeout(reload, 2500); // MJPEG 断开后自动重连
	}

	$effect(() => {
		if (isLocal) openLocal();
		else closeLocal();
		return () => clearTimeout(retryTimer);
	});

	async function doCapture() {
		try {
			const f = await grabFrame();
			if (!f) return;
			if (oncapture) oncapture(f);
		} catch (e) {
			errorText = String(e.message || e);
		}
	}

</script>

<div class="pane {compact ? 'compact' : ''}" bind:this={wrapEl}>
	<div class="pane-head">
		<span class="pane-title">{title}</span>
		<select class="pane-src" bind:value={value}>
			{#if cameras.length === 0}
				<option value="" disabled>（无可用画面源：未连接 Jetson）</option>
			{/if}
			{#each cameras as c (c.id)}
				<option value={c.id}>{c.name}{c.ready ? '' : '（不可达）'}</option>
			{/each}
			<option value="local">本机摄像头（浏览器）</option>
			<option value="custom">自定义 MJPEG 地址</option>
		</select>
	</div>

	<div class="pane-view">
		{#if isLocal}
			<video bind:this={videoEl} autoplay playsinline muted></video>
		{:else if isCustom}
			{#if customUrl}
				<img src={customUrl} alt="自定义 MJPEG 流" onload={onImgLoad} onerror={onImgError} />
			{:else}
				<div class="pane-holder">
					<p>请填写 MJPEG 地址</p>
					<input placeholder="host:port/video 或完整流地址" bind:value={customUrl} />
				</div>
			{/if}
		{:else if cam}
			{#if !cam.ready && !online}
				<div class="pane-holder">
					<p>该路画面不可达</p>
					<p class="sub">{cam.detail}</p>
					{#if cam.error}<p class="sub err">{cam.error}</p>{/if}
					<button class="mini" onclick={reload}>重试</button>
				</div>
			{:else}
				{#key tick}
					<img src={streamUrl} alt={cam.name} onload={onImgLoad} onerror={onImgError} />
				{/key}
			{/if}
		{:else}
			<div class="pane-holder">
				{#if cameras.length === 0}
					<p>未连接 Jetson 服务</p>
					<p class="sub">在下方「连接设置」填写 Jetson 地址后点「连接」，或在左侧选择本机摄像头</p>
				{:else}
					<p>未选择画面源</p>
				{/if}
			</div>
		{/if}

		{#if online || (isLocal && localStream)}
			<span class="badge-live">● LIVE</span>
		{/if}
	</div>

	<div class="pane-foot">
		<span class="pane-status {status.ok ? 'ok' : 'err'}">{status.text}</span>
		{#if errorText}<span class="pane-err">{errorText}</span>{/if}
		<span class="spacer"></span>
		<button class="mini" onclick={doCapture} title="抓拍当前画面">📸</button>
		{#if showDetect}
			<button class="mini accent" onclick={ondetect} disabled={detecting}>
				{detecting ? '识别中…' : '🎯 AI 识别'}
			</button>
		{/if}
		<button class="mini" onclick={reload} title="重连">⟳</button>
		<button class="mini" onclick={fullscreen} title="全屏">⛶</button>
	</div>
</div>

<style>
	.pane {
		display: flex;
		flex-direction: column;
		min-width: 0;
		background: #0b1220;
		border: 1px solid #1f2a3d;
		border-radius: 10px;
		overflow: hidden;
	}
	.pane-head {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 8px;
		background: #111c2e;
		border-bottom: 1px solid #1f2a3d;
	}
	.pane-title {
		font-size: 12px;
		font-weight: 600;
		color: #9fb3c8;
		white-space: nowrap;
	}
	.pane-src {
		flex: 1;
		min-width: 0;
		font-size: 12px;
		padding: 3px 6px;
		border-radius: 6px;
		background: #0b1220;
		color: #dbe6f3;
		border: 1px solid #26364d;
	}
	.pane-view {
		position: relative;
		aspect-ratio: 4 / 3;
		background: #05090f;
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 160px;
	}
	.pane.compact .pane-view {
		min-height: 120px;
	}
	.pane-view img,
	.pane-view video {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
	}
	.pane-holder {
		padding: 12px;
		text-align: center;
		color: #8ea3bb;
		font-size: 13px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		align-items: center;
	}
	.pane-holder input {
		width: 90%;
		padding: 4px 6px;
		border-radius: 6px;
		border: 1px solid #26364d;
		background: #0b1220;
		color: #dbe6f3;
		font-size: 12px;
	}
	.pane-holder .sub {
		font-size: 11px;
		color: #6b7f96;
		margin: 0;
	}
	.pane-holder .err {
		color: #ff9f9f;
	}
	.badge-live {
		position: absolute;
		left: 8px;
		bottom: 8px;
		font-size: 11px;
		color: #7CFFB2;
		background: rgba(0, 0, 0, 0.55);
		padding: 1px 6px;
		border-radius: 5px;
	}
	.pane-foot {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 5px 8px;
		background: #111c2e;
		border-top: 1px solid #1f2a3d;
		font-size: 11px;
		flex-wrap: wrap;
	}
	.pane-status.ok {
		color: #7fe0a8;
	}
	.pane-status.err {
		color: #ffb1b1;
	}
	.pane-err {
		color: #ffcf8a;
	}
	.spacer {
		flex: 1;
	}
	.mini {
		border: 1px solid #26364d;
		background: #16233a;
		color: #cfe0f5;
		border-radius: 6px;
		padding: 2px 7px;
		font-size: 12px;
		cursor: pointer;
	}
	.mini:hover {
		background: #1d2d49;
	}
	.mini.accent {
		background: #0e7490;
		border-color: #0e7490;
		color: #fff;
	}
	.mini:disabled {
		opacity: 0.6;
		cursor: default;
	}
</style>

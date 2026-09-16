<script>
  import { onMount, onDestroy, tick } from 'svelte';
  import jsQR from 'jsqr';

  // ===================== 显示 / 轮次状态 =====================
  let display = $state({
    sequence: [], counts: {}, total: 0, current: null,
    display_locked: false, unlock_at: 0, hold_seconds: 3,
    bins: [], round: { status: 'idle', started_at: null, elapsed: 0, reason: null },
    constants: { bin_count: 6, bin_capacity: 4, tray_size_mm: 160, goods_size_mm: 40 },
    shapes: [], colors: [], marks: ['无', '污渍', '缺陷']
  });
  let goods = $state([]);
  let formats = $state(['svg', 'png', 'jpg', 'webp', 'gif', 'bmp']);
  let imgFmt = $state('svg');
  let nowTick = $state(Date.now() / 1000);
  let status = $state({ gateway: 'unknown', robot_state: 'idle', intervention_allowed: true });
  let debug = $state(false);            // 调试模式：比赛进行中也允许联调干预
  let toast = $state(null);
  let toastTimer = null;

  // ---- 自动分拣 / 视觉容器（Jetson 端运行，PC 端仅显示与控制）----
  let auto = $state({ status: 'idle', count: 0, error: null, dry_run: true, configured: true,
                      waiting_clear: false, last_marks: null, last_event: null, last_robot: null,
                      vision_url: null });
  let autoBusy = $state('');
  let autoMaxItems = $state(0);          // 0 = 不限件数
  let autoDryRun = $state(true);
  let autoRequireAck = $state(false);
  let camSrc = $state('local');          // local(本机摄像头) | vision(Jetson 视觉容器画面)
  let frameTick = $state(0);
  let visionOk = $state(false);
  let marks = $state({ detections: [], qr_text: '', ts: 0, error: null, last_event: null });

  // ---- 实际服务连接状态（以真实部署为准，PC 容器本身不提供任何服务）----
  let deploy = $state({ checked: false, deployed: false, gateway_url: null, vision_url: null,
                        gateway_reachable: false, role: '', vision: null });

  const ROBOT_LABEL = { idle: '空闲', working: '分拣中', paused: '已暂停', reset: '复位中', estop: '紧急停止' };
  const remain = $derived(Math.max(0, (display.unlock_at || 0) - nowTick));
  const round = $derived(display.round || {});
  const running = $derived(round.status === 'running');
  const finished = $derived(round.status === 'finished');
  const elapsed = $derived(
    running && round.started_at ? Math.max(0, nowTick - round.started_at) : round.elapsed || 0
  );
  const tableRows = $derived([...display.sequence].reverse());
  const bins = $derived(display.bins || []);
  const usedBins = $derived(bins.filter((b) => !b.empty).length);
  const totalInBins = $derived(bins.reduce((n, b) => n + b.count, 0));

  function fmtTime(sec) {
    const s = Math.max(0, sec || 0);
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`;
  }

  function showToast(text, kind = 'ok') {
    toast = { text, kind };
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 3600);
  }
  async function api(path, opts = {}) {
    const resp = await fetch(path, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const err = new Error(data.detail || `HTTP ${resp.status}`);
      err.status = resp.status;
      throw err;
    }
    return data;
  }
  const jpost = (path, body) => api(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {})
  });
  const debugQ = (p) => (debug ? `${p}${p.includes('?') ? '&' : '?'}debug=1` : p);

  // ===================== 轮询 =====================
  async function refreshDisplay() {
    try {
      const d = await api('/api/display');
      display = d;
      if (d.goods?.length) goods = d.goods;
      if (d.formats?.length) formats = d.formats;
    } catch { /* 网关未就绪 */ }
  }
  async function refreshStatus() {
    try { status = await api('/api/status'); } catch { status = { ...status, gateway: 'offline' }; }
  }
  async function refreshAuto() {
    try {
      const d = await api('/api/auto/status');
      if (d.configured === false) { auto = { ...auto, status: 'unconfigured', configured: false }; return; }
      auto = { ...auto, ...d, configured: true };
    } catch (e) {
      auto = { ...auto, configured: e.status !== 501, error: e.status === 501 ? null : e.message };
    }
  }
  async function refreshMarks() {
    try {
      const d = await api('/api/marks/latest');
      marks = { ...marks, ...d, error: null };
    } catch (e) {
      marks = { ...marks, detections: [], error: e.status === 501 ? '未配置视觉容器' : e.message };
    }
  }
  async function refreshVision() {
    try { const d = await api('/api/vision/health'); visionOk = d.ok !== false; }
    catch { visionOk = false; }
  }
  async function refreshHealth() {
    try {
      const d = await api('/health');
      deploy = { checked: true, ...d };
    } catch (e) {
      deploy = { ...deploy, checked: true, gateway_reachable: false };
    }
  }
  /** 服务未连接时统一提示（避免展示与真实部署不符的数据） */
  function notDeployedMsg(what) {
    return deploy.gateway_url
      ? `${what}不可达（${deploy.gateway_url}）—— 请检查 Jetson 端服务是否已启动`
      : `未连接实际服务：PC 容器不含 ${what}，请在 .env 配置 GATEWAY_URL / VISION_URL 指向 Jetson`;
  }
  async function startAuto() {
    autoBusy = 'start';
    try {
      await jpost('/api/auto/start', {
        max_items: Number(autoMaxItems) || 0, dry_run: autoDryRun,
        require_ack: autoRequireAck, auto_start_round: true
      });
      showToast('自动分拣已启动（识别 → 规则入库 → 显示 → 下一件）');
      await refreshAuto();
    } catch (e) { showToast(e.status === 501 ? 'PC 容器未配置 Jetson 网关（GATEWAY_URL）' : e.message, 'err'); }
    finally { autoBusy = ''; }
  }
  async function stopAuto() {
    autoBusy = 'stop';
    try { await jpost('/api/auto/stop'); showToast('自动分拣已停止'); await refreshAuto(); }
    catch (e) { showToast(e.message, 'err'); } finally { autoBusy = ''; }
  }
  let pollTimer, tickTimer, clockTimer, autoTimer, marksTimer, frameTimer, healthTimer;
  onMount(() => {
    refreshHealth();
    refreshDisplay(); refreshStatus(); refreshTasks();
    refreshAuto(); refreshMarks(); refreshVision();
    pollTimer = setInterval(() => { refreshDisplay(); refreshStatus(); }, 900);
    autoTimer = setInterval(refreshAuto, 1000);
    marksTimer = setInterval(() => { refreshMarks(); refreshVision(); }, 1500);
    healthTimer = setInterval(refreshHealth, 5000);
    frameTimer = setInterval(() => (frameTick = Date.now()), 1200);   // 视觉画面刷新节拍
    tickTimer = setInterval(() => (nowTick = Date.now() / 1000), 100);
    const upd = () => (clock.t = new Date().toLocaleTimeString('zh-CN', { hour12: false }));
    upd(); clockTimer = setInterval(upd, 1000);
  });
  onDestroy(() => {
    clearInterval(pollTimer); clearInterval(tickTimer); clearInterval(clockTimer); clearTimeout(toastTimer);
    clearInterval(autoTimer); clearInterval(marksTimer); clearInterval(frameTimer); clearInterval(healthTimer);
    if (cameraStream) cameraStream.getTracks().forEach((t) => t.stop());
    stopQrScanning();
  });

  // ===================== 摄像头 / 识别 =====================
  let cameraActive = $state(false);
  let cameraStream = null;
  let videoEl = $state(undefined);
  let canvasEl = $state(undefined);
  let snapshotEl;
  let snapshotUrl = $state(null);
  let mjpegUrl = $state('');
  let camMode = $state('browser');
  let detections = $state([]);
  let detecting = $state(false);
  let detectInfo = $state('');
  let detectEngine = $state('');
  let qrText = $state('');
  let selectedShape = $state('五棱柱');
  let selectedColor = $state('青色');
  let selectedMark = $state('无');
  let itemText = $state('');
  let itemQr = $state('');

  async function openCamera() {
    if (cameraStream) return;
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
      });
      cameraActive = true; camMode = 'browser';
      if (videoEl) videoEl.srcObject = cameraStream;
    } catch (e) { showToast('无法访问摄像头：' + (e.message || e), 'err'); }
  }
  function closeCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
    cameraActive = false; snapshotUrl = null; detections = [];
  }
  function applyMjpeg() {
    if (!/^https?:\/\//.test(mjpegUrl)) { showToast('请输入有效 MJPEG 流地址', 'err'); return; }
    if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
    camMode = 'mjpeg'; cameraActive = true; showToast('已接入 MJPEG 流');
  }
  function capture() {
    if (camMode === 'browser' && videoEl?.videoWidth) {
      snapshotEl = snapshotEl || document.createElement('canvas');
      snapshotEl.width = videoEl.videoWidth; snapshotEl.height = videoEl.videoHeight;
      snapshotEl.getContext('2d').drawImage(videoEl, 0, 0);
      snapshotUrl = snapshotEl.toDataURL('image/jpeg', 0.88);
      return true;
    }
    showToast('请先打开本机摄像头再抓拍', 'warn');
    return false;
  }
  async function runDetect() {
    if (!snapshotUrl && !capture()) return;
    detecting = true; detectInfo = 'AI 识别中…';
    try {
      const blob = await (await fetch(snapshotUrl)).blob();
      const form = new FormData();
      form.append('file', blob, 'snapshot.jpg');
      const d = await api('/api/detect', { method: 'POST', body: form });
      detections = d.detections || [];
      detectEngine = d.engine; qrText = d.qr_text || '';
      detectInfo = d.message || `识别到 ${detections.length} 件`;
      await drawBoxes();
      if (!detections.length) showToast('未识别到货物，请调整视角/光照', 'warn');
    } catch (e) {
      if (e.status === 501) {
        detectInfo = e.message; detections = [];
        showToast('识别在机器人端完成，可用下方「登记分拣」写入', 'warn');
      } else { detectInfo = '识别失败：' + e.message; showToast(detectInfo, 'err'); }
    } finally { detecting = false; }
  }
  async function drawBoxes() {
    await tick();
    if (!snapshotUrl || !canvasEl) return;
    const img = new Image();
    img.onload = () => {
      canvasEl.width = img.width; canvasEl.height = img.height;
      const ctx = canvasEl.getContext('2d');
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      for (const d of detections) {
        const x = d.x * img.width, y = d.y * img.height, w = d.w * img.width, h = d.h * img.height;
        ctx.strokeStyle = 'rgba(8,145,178,.95)';
        ctx.lineWidth = Math.max(2, img.width / 420);
        ctx.strokeRect(x, y, w, h);
        const label = `${d.name || d.label} ${((d.conf || 0) * 100).toFixed(0)}%`;
        ctx.font = `${Math.max(14, img.width / 42)}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(8,145,178,.92)';
        ctx.fillRect(x, Math.max(0, y - 24), tw + 14, 24);
        ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
        ctx.fillText(label, x + 7, Math.max(0, y - 24) + 12);
      }
    };
    img.src = snapshotUrl;
  }

  // ===================== 分拣登记 / 轮次控制 =====================
  async function registerItem() {
    const g = goods.find((x) => x.shape === selectedShape && x.color === selectedColor);
    try {
      const r = await jpost(debugQ('/api/sort/event'), {
        class: g?.id, shape: selectedShape, color: selectedColor,
        marks: selectedMark === '无' ? [] : [selectedMark],
        text: itemText, qr: itemQr, source: 'manual', image_format: imgFmt
      });
      showToast(`已登记：${r.record.name} → ${r.record.bin} 号储物盒（第 ${r.record.seq} 件）`);
      itemText = ''; itemQr = '';
      await refreshDisplay();
    } catch (e) { showToast(e.message, 'err'); }
  }
  async function registerTop() {
    const top = detections[0];
    if (!top) { showToast('请先执行 AI 识别', 'warn'); return; }
    try {
      const r = await jpost(debugQ('/api/sort/event'), {
        class: top.class, shape: top.shape, color: top.color, marks: top.marks || [],
        text: top.text, qr: top.qr, confidence: top.conf, source: 'vision', image_format: imgFmt
      });
      showToast(`已记录：${r.record.name} → ${r.record.bin} 号储物盒`);
      await refreshDisplay();
    } catch (e) { showToast(e.message, 'err'); }
  }
  async function simulateSort() {
    try {
      const r = await jpost(debugQ('/api/sort/demo'));
      showToast(`模拟分拣：${r.record.name} → ${r.record.bin} 号盒（第 ${r.record.seq} 件）`);
      await refreshDisplay();
    } catch (e) { showToast(e.message, 'err'); }
  }
  async function ackDisplay() {
    try { await jpost('/api/sort/ack'); showToast('已确认显示，可投放下一件'); await refreshDisplay(); }
    catch (e) { showToast(e.message, 'err'); }
  }
  async function resetAll() {
    try { await jpost(debugQ('/api/sort/reset')); showToast('已复位（显示屏 + 六个储物盒 + 轮次）'); await refreshDisplay(); }
    catch (e) { showToast(e.message, 'err'); }
  }
  let busy = $state('');
  async function startRound() {
    busy = 'start';
    try { await jpost('/api/round/start'); showToast('比赛开始，计时启动'); await refreshDisplay(); }
    catch (e) { showToast(e.message, 'err'); } finally { busy = ''; }
  }
  async function stopRound(reason) {
    busy = reason;
    try {
      const r = await jpost('/api/round/stop', { reason });
      showToast(`${r.round.reason}（用时 ${fmtTime(r.round.elapsed_ms / 1000)}）`);
      await refreshDisplay();
    } catch (e) { showToast(e.message, 'err'); } finally { busy = ''; }
  }
  async function reportDrop(where) {
    try {
      const r = await jpost('/api/round/fault', { where });
      showToast(`${r.round.reason}`, 'err');
      await refreshDisplay();
    } catch (e) { showToast(e.message, 'err'); }
  }
  async function robotCommand(action) {
    busy = action;
    try { const d = await jpost('/api/robot/action', { action }); showToast(d.message); await refreshStatus(); }
    catch (e) { showToast('指令失败：' + e.message, 'err'); } finally { busy = ''; }
  }
  const imgUrl = (id, fmt) => `/api/images/${id}.${fmt || imgFmt}`;
  const hasFmt = (id, fmt) => Boolean(goods.find((g) => g.id === id)?.images?.[fmt || imgFmt]);

  // ===================== 扫码领取任务 =====================
  let qrOpen = $state(false), qrStream = null, qrVideoEl = $state(undefined), qrCanvas, qrTimer = null;
  let qrRaw = $state(''), qrCode = $state(''), qrError = $state(''), showDemoQr = $state(false), demoList = $state([]);
  let worker = $state(localStorage.getItem('sort_worker') || '操作员-01');
  let tasks = $state([]), currentTask = $state(null);
  const clock = $state({ t: '' });

  async function refreshTasks() {
    try { const d = await api('/api/tasks'); tasks = d.tasks || []; currentTask = d.current || null; } catch {}
  }
  function openQr() { qrOpen = true; qrRaw = ''; qrCode = ''; qrError = ''; setTimeout(startQrScan, 60); }
  function closeQr() { qrOpen = false; stopQrScanning(); }
  async function startQrScan() {
    try {
      qrStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 960 }, height: { ideal: 640 } }, audio: false
      });
      if (qrVideoEl) qrVideoEl.srcObject = qrStream;
      qrTimer = setInterval(scanFrame, 180);
    } catch (e) { qrError = '无法打开扫描摄像头：' + (e.message || e); }
  }
  function stopQrScanning() {
    if (qrTimer) { clearInterval(qrTimer); qrTimer = null; }
    if (qrStream) { qrStream.getTracks().forEach((t) => t.stop()); qrStream = null; }
  }
  function scanFrame() {
    const v = qrVideoEl;
    if (!v?.videoWidth || qrRaw) return;
    qrCanvas = qrCanvas || document.createElement('canvas');
    const scale = Math.min(1, 640 / v.videoWidth);
    const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
    qrCanvas.width = w; qrCanvas.height = h;
    const ctx = qrCanvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, w, h);
    const { data } = ctx.getImageData(0, 0, w, h);
    const code = jsQR(data, w, h, { inversionAttempts: 'dontInvert' });
    if (code?.data) {
      qrRaw = code.data.trim();
      const m = qrRaw.match(/SORT-TASK:?[:\s]*(T-?[\w-]+)/i) || qrRaw.match(/(T-?[\w-]{4,})/i);
      qrCode = m ? m[1].toUpperCase() : qrRaw;
      stopQrScanning();
    }
  }
  async function claim(taskId) {
    const id = taskId || qrCode;
    if (!id) { showToast('请扫描或输入任务码', 'warn'); return; }
    try {
      const d = await jpost('/api/tasks/claim', { task_id: id, worker });
      localStorage.setItem('sort_worker', worker);
      currentTask = d.task; closeQr(); await refreshTasks(); await refreshDisplay();
      showToast(`已领取任务 ${d.task.id}，显示屏与储物盒已复位`);
    } catch (e) { showToast('领取失败：' + e.message, 'err'); }
  }
  async function finishTask(flag) {
    if (!currentTask) return;
    try {
      await jpost(`/api/tasks/${currentTask.id}/${flag}`);
      currentTask = null; await refreshTasks();
      showToast(flag === 'complete' ? '任务完成 ✅' : '已放弃任务');
    } catch (e) { showToast(e.message, 'err'); }
  }
  async function toggleDemoQr() {
    if (!demoList.length) {
      try { demoList = (await api('/api/tasks?available=1')).tasks.slice(0, 6); } catch (e) { showToast(e.message, 'err'); return; }
    }
    showDemoQr = !showDemoQr;
  }
  const demoQrUrl = (id) => `https://api.qrserver.com/v1/create-qr-code/?size=140x140&data=${encodeURIComponent('SORT-TASK:' + id)}&bgcolor=ffffff&color=0f172a&margin=6`;

  $effect(() => {
    if (videoEl && cameraStream && cameraActive) videoEl.srcObject = cameraStream;
    if (qrVideoEl && qrStream && qrOpen) qrVideoEl.srcObject = qrStream;
  });
</script>

<svelte:head>
  <title>智能分拣装置 · 分拣信息显示屏</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

<main class="shell">
  <!-- ============ 顶栏 ============ -->
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">◆</div>
      <div>
        <h1>智能分拣装置 · 分拣信息显示屏</h1>
        <p>六个储物盒（1–6）· 同形同色同盒 · 每盒≤{display.constants.bin_capacity} 件 · 一次一件 · 显示完成再分拣下一件</p>
      </div>
    </div>
    <div class="statusbar">
      <span class="pill">
        <i class="dot {deploy.gateway_reachable ? 'ok' : deploy.gateway_url ? 'err' : 'idle'}"></i>
        {deploy.gateway_reachable ? '实际服务已连接' : deploy.gateway_url ? '服务不可达' : '未部署服务'}
      </span>
      <span class="pill"><i class="dot {status.robot_state === 'working' ? 'ok' : status.robot_state === 'estop' ? 'err' : 'idle'}"></i>机械臂 {ROBOT_LABEL[status.robot_state] || status.robot_state || '—'}</span>
      <span class="pill"><i class="dot {running ? 'warn' : finished ? 'err' : 'idle'}"></i>
        {running ? '比赛进行中' : finished ? '本轮已结束' : '待机'}
      </span>
      <span class="pill timer">⏱ {fmtTime(elapsed)}</span>
      <span class="pill">累计 <b>{display.total}</b> 件</span>
      <span class="pill clock">{clock.t}</span>
    </div>
  </header>

  <!-- ============ 服务连接状态（以实际服务为准）============ -->
  {#if deploy.checked && !deploy.gateway_reachable}
    <div class="banner offline">
      <span class="banner-icon">⚠</span>
      <span class="banner-text">
        <b>未连接实际服务</b> —— PC 调试容器只提供前端页面，不包含分拣/识别/自动分拣服务。
        {#if deploy.gateway_url}
          已配置 <code>GATEWAY_URL={deploy.gateway_url}</code>，但服务不可达（请确认 Jetson 端容器已启动）。
        {:else}
          请在 <code>sort-web/.env</code> 中填写 <code>GATEWAY_URL=http://&lt;jetson-ip&gt;</code>（视觉容器可另配
          <code>VISION_URL=http://&lt;jetson-ip&gt;:8100</code>），然后 <code>docker compose up -d</code> 重启本容器。
        {/if}
        页面上所有数据均来自实际服务；未连接时显示为“无数据”。
      </span>
    </div>
  {/if}

  <!-- ============ 赛项状态条 ============ -->
  <div class="banner {running ? 'running' : finished ? 'finished' : 'ready'}">
    {#if running}
      <span class="banner-icon">⏱</span>
      <span class="banner-text">
        比赛进行中 · 计时 {fmtTime(elapsed)} —— 参赛队员不得接触装置
        {#if display.display_locked}｜本件分拣信息显示中（{remain.toFixed(1)}s）{/if}
      </span>
      <button class="btn ghost sm" onclick={() => stopRound('complete')} disabled={busy === 'complete'}>本轮完成</button>
      <button class="btn sm danger" onclick={() => stopRound('manual')} disabled={busy === 'manual'}>结束本轮</button>
    {:else if finished}
      <span class="banner-icon">🏁</span>
      <span class="banner-text">本轮结束：{round.reason || '已停止'} —— 总用时 {fmtTime(elapsed)}，分拣成功 {display.total} 件</span>
      <button class="btn accent sm" onclick={resetAll}>复位新一轮</button>
    {:else if display.display_locked}
      <span class="banner-icon">⏳</span>
      <span class="banner-text">分拣信息显示中 —— 请勿分拣下一件货物（剩余 {remain.toFixed(1)}s）</span>
      <button class="btn accent sm" onclick={ackDisplay}>确认显示，立即放行</button>
    {:else}
      <span class="banner-icon">🎫</span>
      <span class="banner-text">待机中 —— 扫码领取分拣任务后，按统一指令启动装置（计时开始）</span>
      <button class="btn primary sm" onclick={startRound} disabled={busy === 'start'}>▶ 开始比赛</button>
      <button class="btn ghost sm" onclick={resetAll}>复位</button>
    {/if}
  </div>

  <div class="layout">
    <!-- ============ 左：摄像头 + 识别 + 快捷任务 ============ -->
    <section class="col">
      <div class="card">
        <div class="card-head">
          <h2>📷 摄像头画面 · AI 识别</h2>
          <div class="src-switch">
            <button class="src-btn {camSrc === 'local' ? 'active' : ''}" onclick={() => (camSrc = 'local')}>本机摄像头</button>
            <button class="src-btn {camSrc === 'vision' ? 'active' : ''}" onclick={() => (camSrc = 'vision')}>
              视觉画面(Jetson)<i class="src-dot {visionOk ? 'ok' : 'err'}"></i>
            </button>
          </div>
          <div class="cam-tools">
            <input class="mjpeg-input" placeholder="MJPEG 流 http://ip:8080/video" bind:value={mjpegUrl} />
            <button class="btn ghost sm" onclick={applyMjpeg}>接入流</button>
          </div>
        </div>
        <div class="cam-view">
          {#if camSrc === 'vision'}
            {#if visionOk}
              <img src="/api/frame.jpg?draw=1&t={frameTick}" alt="视觉容器画面（含标记）" />
              <span class="cam-label">● VISION</span>
              <span class="snap-badge">识别标记已叠加</span>
            {:else}
              <div class="cam-holder">
                <div class="cam-icon">🛰</div>
                <p>未连接视觉容器</p>
                <p class="sub">
                  {deploy.vision_url ? `已配置 ${deploy.vision_url}，但不可达` : '请在 .env 配置 VISION_URL 指向 Jetson 视觉容器（:8100）'}
                </p>
                <p class="sub">画面与标记均来自实际服务，未连接时不展示模拟内容</p>
              </div>
            {/if}
          {:else if cameraActive && camMode === 'browser'}
            <video bind:this={videoEl} autoplay playsinline muted></video>
          {:else if cameraActive && camMode === 'mjpeg'}
            <img src={mjpegUrl} alt="MJPEG" />
          {:else}
            <div class="cam-holder">
              <div class="cam-icon">📷</div>
              <p>摄像头未开启</p>
              <p class="sub">点击下方「打开摄像头」，或切到「视觉画面」查看 Jetson 端识别画面</p>
            </div>
          {/if}
          {#if camSrc === 'local' && snapshotUrl}
            <img class="snapshot" src={snapshotUrl} alt="抓拍" />
            <canvas bind:this={canvasEl} class="overlay"></canvas>
          {/if}
          {#if camSrc === 'local'}<span class="cam-label">● LIVE</span>{/if}
        </div>
        <div class="cam-actions">
          {#if camSrc === 'local'}
            <button class="btn primary" onclick={() => (cameraActive ? closeCamera() : openCamera())}>
              {cameraActive ? '⏻ 关闭摄像头' : '📷 打开摄像头'}
            </button>
            <button class="btn" onclick={capture} disabled={!cameraActive}>📸 抓拍</button>
            <button class="btn" onclick={runDetect} disabled={detecting}>{detecting ? '识别中…' : '🎯 AI 识别'}</button>
            <button class="btn accent" onclick={registerTop} disabled={!detections.length}>✅ 识别结果登记</button>
          {:else}
            <span class="hint">
              画面来自视觉容器（Jetson :8100），标记由识别引擎输出
              {#if marks.ts}· 更新于 {new Date(marks.ts * 1000).toLocaleTimeString('zh-CN', { hour12: false })}{/if}
              {#if marks.error}· <b class="warn-text">{marks.error}</b>{/if}
            </span>
          {/if}
          <span class="hint">{detectInfo}</span>
        </div>

        {#if marks.detections?.length || marks.qr_text}
          <div class="ai-result">
            <div class="ai-head">
              <span class="ai-badge">标记输出（视觉容器）· {marks.detections.length} 件</span>
              <span class="ai-note">颜色 · 形状 · 污渍/缺陷 · 二维码/文字 · 检测框</span>
            </div>
            {#each marks.detections as d, i}
              <div class="ai-row">
                <span class="ai-idx">#{i + 1}</span>
                <span class="ai-name">{d.name}</span>
                <span class="tag">颜色 {d.color}</span>
                <span class="tag">形状 {d.shape}</span>
                <span class="tag">表面 {(d.marks || ['无']).join('·')}</span>
                <span class="tag">顶点 {d.vertices} · 圆度 {d.circularity}</span>
                <span class="tag conf">置信 {((d.conf || 0) * 100).toFixed(0)}%</span>
                <span class="tag box">框 [{d.box?.join(', ')}]</span>
              </div>
            {/each}
            {#if marks.qr_text}<div class="ai-qr">🔳 二维码 / 文字：<b>{marks.qr_text}</b></div>{/if}
          </div>
        {/if}
      </div>

      <!-- 自动分拣（循环运行在 Jetson 网关容器内） -->
      <div class="card auto-card">
        <div class="card-head">
          <h2>🤖 自动分拣</h2>
          <span class="hint">
            {#if !auto.configured}PC 调试容器未配置 Jetson 网关（GATEWAY_URL）
            {:else}视觉 {auto.vision_url || '—'} · 机器人 {auto.last_robot?.mode || '—'}{/if}
          </span>
        </div>
        <div class="auto-body">
          <div class="auto-state {!deploy.gateway_reachable ? 'err' : auto.status === 'running' ? 'run' : auto.status === 'error' ? 'err' : auto.status === 'finished' ? 'done' : ''}">
            <span class="auto-state-dot"></span>
            {#if !deploy.gateway_reachable}
              未连接实际服务
            {:else if auto.status === 'running'}运行中
            {:else if auto.status === 'error'}异常
            {:else if auto.status === 'finished'}已完成
            {:else if auto.status === 'stopped'}已停止
            {:else}待机{/if}
          </div>
          <div class="auto-metrics">
            <div class="metric"><span class="metric-num">{deploy.gateway_reachable ? (auto.count || 0) : '—'}</span><span class="metric-label">自动分拣件数</span></div>
            <div class="metric"><span class="metric-num small">{!deploy.gateway_reachable ? '无数据' : auto.waiting_clear ? '等待取走' : (auto.last_marks?.top?.name || '空托盘')}</span><span class="metric-label">托盘状态</span></div>
            <div class="metric"><span class="metric-num small">{!deploy.gateway_reachable ? '无数据' : auto.last_event ? `#${auto.last_event.seq} → ${auto.last_event.bin}号盒` : '—'}</span><span class="metric-label">最近入库</span></div>
          </div>
          {#if !deploy.gateway_reachable}
            <div class="auto-err">{notDeployedMsg('分拣网关')}</div>
          {:else if auto.error}
            <div class="auto-err">{auto.error}</div>
          {/if}
          <div class="auto-controls">
            <label class="mini-field">件数上限
              <input type="number" min="0" bind:value={autoMaxItems} placeholder="0=不限" />
            </label>
            <label class="chk"><input type="checkbox" bind:checked={autoDryRun} /> 仅记录（不驱动机械臂）</label>
            <label class="chk"><input type="checkbox" bind:checked={autoRequireAck} /> 需人工确认显示</label>
            {#if auto.status === 'running' && deploy.gateway_reachable}
              <button class="btn danger" onclick={stopAuto} disabled={autoBusy === 'stop'}>■ 停止自动分拣</button>
            {:else}
              <button class="btn accent" onclick={startAuto} disabled={autoBusy === 'start' || !deploy.gateway_reachable}>
                ▶ 开始自动分拣（自动开始本轮）
              </button>
            {/if}
          </div>
          <div class="auto-flow">
            识别（视觉容器）→ 连续 3~5 帧确认 → 同形同色同盒入库 → 显示屏保持 3s → 机械臂取放 → 托盘空 → 下一件
          </div>
        </div>
      </div>

      <!-- 快捷任务 -->
      <div class="card">
        <div class="card-head">
          <h2>⚡ 快捷任务</h2>
          <label class="debug-toggle" title="比赛进行中如需联调，勾选后允许人工干预">
            <input type="checkbox" bind:checked={debug} /> 调试模式
          </label>
        </div>
        <div class="quick-grid">
          <button class="qbtn" style="--c:#7c3aed" onclick={openQr}>
            <span class="qicon">⌘</span><span class="qtext">扫码领取分拣任务</span><span class="qdesc">扫描现场任务二维码</span>
          </button>
          <button class="qbtn" style="--c:#16a34a" onclick={startRound} disabled={busy === 'start' || running}>
            <span class="qicon">▶</span><span class="qtext">开始比赛</span><span class="qdesc">统一指令启动 · 计时开始</span>
          </button>
          <button class="qbtn" style="--c:#0d9488" onclick={ackDisplay} disabled={!display.display_locked}>
            <span class="qicon">✅</span><span class="qtext">确认显示完成</span><span class="qdesc">放行下一件货物</span>
          </button>
          <button class="qbtn" style="--c:#ca8a04" onclick={() => stopRound('complete')} disabled={!running}>
            <span class="qicon">🏁</span><span class="qtext">本轮完成</span><span class="qdesc">停止计时</span>
          </button>
          <button class="qbtn danger" style="--c:#dc2626" onclick={() => reportDrop('in')}>
            <span class="qicon">⚠</span><span class="qtext">掉落（装置内）</span><span class="qdesc">本轮比赛结束</span>
          </button>
          <button class="qbtn danger" style="--c:#b91c1c" onclick={() => reportDrop('out')}>
            <span class="qicon">⚠</span><span class="qtext">掉落（装置外）</span><span class="qdesc">本轮比赛结束</span>
          </button>
          <button class="qbtn" style="--c:#0284c7" onclick={() => robotCommand('pause')} disabled={busy === 'pause'}>
            <span class="qicon">⏸</span><span class="qtext">暂停</span><span class="qdesc">暂停当前流程</span>
          </button>
          <button class="qbtn" style="--c:#64748b" onclick={() => robotCommand('reset')} disabled={busy === 'reset'}>
            <span class="qicon">🔄</span><span class="qtext">复位机械臂</span><span class="qdesc">回到安全位姿</span>
          </button>
          <button class="qbtn" style="--c:#dc2626" onclick={() => robotCommand('estop')} disabled={busy === 'estop'}>
            <span class="qicon">⛔</span><span class="qtext">紧急停止</span><span class="qdesc">立即停机断电</span>
          </button>
        </div>

        <!-- 联调：登记一件分拣（比赛进行中需调试模式） -->
        <div class="manual-row">
          <span class="manual-label">登记分拣</span>
          <select bind:value={selectedShape}>
            {#each display.shapes as s}<option value={s}>{s}</option>{/each}
          </select>
          <select bind:value={selectedColor}>
            {#each display.colors as c}<option value={c}>{c}</option>{/each}
          </select>
          <select bind:value={selectedMark}>
            {#each display.marks as m}<option value={m}>{m === '无' ? '无污渍缺陷' : m}</option>{/each}
          </select>
          <input class="mini-input" placeholder="文字(可选)" bind:value={itemText} />
          <input class="mini-input" placeholder="二维码(可选)" bind:value={itemQr} />
          <button class="btn accent sm" onclick={registerItem}>＋ 登记一件</button>
          <button class="btn ghost sm" onclick={simulateSort}>模拟分拣</button>
          <button class="btn ghost sm" onclick={resetAll}>复位</button>
          {#if !status.intervention_allowed}<span class="manual-tip">比赛进行中已锁定，勾选「调试模式」可联调</span>{/if}
        </div>
      </div>
    </section>

    <!-- ============ 中：当前分拣信息 ============ -->
    <section class="col">
      <div class="card current-card">
        <div class="card-head"><h2>🖥 当前分拣信息</h2><span class="hint">显示保持 {display.hold_seconds}s</span></div>
        {#if display.current}
          <div class="cur-grid">
            <div class="cur-seq">
              <span class="cur-seq-label">投放顺序</span>
              <span class="cur-seq-num">#{display.current.seq}</span>
            </div>
            <div class="cur-img-box">
              {#if display.current.class}
                <img src={imgUrl(display.current.class)} alt={display.current.name}
                     onerror={(e) => (e.currentTarget.style.visibility = 'hidden')} />
              {:else}<div class="cur-img-ph">无图片</div>{/if}
              <span class="fmt-badge">
                {display.current.class && hasFmt(display.current.class) ? imgFmt.toUpperCase() : 'SVG'}
              </span>
            </div>
            <div class="cur-info">
              <div class="cur-name">{display.current.name}</div>
              <div class="cur-tags">
                <span class="tag">颜色 {display.current.color}</span>
                <span class="tag">形状 {display.current.shape}</span>
                <span class="tag">储物盒 <b>{display.current.bin} 号</b></span>
                <span class="tag">表面 {(display.current.marks || ['无']).join('·')}</span>
              </div>
              {#if display.current.text}<div class="cur-extra">文字：{display.current.text}</div>{/if}
              {#if display.current.qr}<div class="cur-extra">二维码：{display.current.qr}</div>{/if}
              <div class="cur-total">
                <span class="cur-total-label">分拣成功总数量</span>
                <span class="cur-total-num">{display.current.cumulative}</span>
              </div>
              <div class="cur-ts">{display.current.ts} · 来源 {display.current.source} · 计时 {fmtTime((display.current.elapsed_ms || 0) / 1000)}</div>
            </div>
          </div>
          <div class="cur-state {display.display_locked ? 'locked' : 'ready'}">
            {display.display_locked ? `显示中，${remain.toFixed(1)}s 后可继续` : '已显示完成，可投放下一件'}
          </div>
        {:else}
          <div class="empty">
            <div class="empty-icon">📦</div>
            <p>{deploy.gateway_reachable ? '暂无分拣记录' : '无数据（未连接实际服务）'}</p>
            <p class="sub">
              {#if deploy.gateway_reachable}
                扫码领取任务 → 开始比赛 → 分拣第一件货物，此处将显示投放顺序 / 名称 / 图片 / 成功总数量
              {:else}
                {notDeployedMsg('分拣网关')}
              {/if}
            </p>
          </div>
        {/if}
      </div>

      <!-- 六个储物盒 -->
      <div class="card">
        <div class="card-head">
          <h2>🗄 储物盒（1–{display.constants.bin_count}）</h2>
          <span class="hint">
            {#if deploy.gateway_reachable}
              已用 {usedBins}/{display.constants.bin_count} 盒 · 在盒 {totalInBins} 件 · 每盒≤{display.constants.bin_capacity}
            {:else}无数据（未连接实际服务）{/if}
          </span>
        </div>
        <div class="bin-grid">
          {#each bins as b}
            <div class="bin {b.empty ? 'empty' : b.full ? 'full' : 'used'}">
              <div class="bin-no">{b.no}<span class="bin-unit">号</span></div>
              {#if b.empty}
                <div class="bin-key">空盒</div>
                <div class="bin-bar"><span style="width:0%"></span></div>
                <div class="bin-count">0 / {display.constants.bin_capacity}</div>
              {:else}
                <div class="bin-key">{b.color}{b.shape}</div>
                <div class="bin-thumbs">
                  {#each b.items as it}
                    <img src={it.image} alt={it.name} title={`#${it.seq} ${it.name}`}
                         onerror={(e) => (e.currentTarget.style.visibility = 'hidden')} />
                  {/each}
                </div>
                <div class="bin-bar"><span style="width:{(b.count / display.constants.bin_capacity) * 100}%"></span></div>
                <div class="bin-count {b.full ? 'full' : ''}">{b.count} / {display.constants.bin_capacity}{b.full ? ' 满' : ''}</div>
              {/if}
            </div>
          {/each}
        </div>
        <div class="bin-rule">规则：形状相同且颜色相同的货物分拣到同一个储物盒；每盒最多 {display.constants.bin_capacity} 件（托盘 {display.constants.tray_size_mm}×{display.constants.tray_size_mm}mm，货物 ≤{display.constants.goods_size_mm}mm）</div>
      </div>
    </section>

    <!-- ============ 右：图库 + 任务 ============ -->
    <section class="col side">
      <div class="card">
        <div class="card-head">
          <h2>🖼 货物图库 · 多格式</h2>
          <div class="fmt-switch">
            {#each formats as f}
              <button class="fmt-btn {imgFmt === f ? 'active' : ''}" onclick={() => (imgFmt = f)}>{f}</button>
            {/each}
          </div>
        </div>
        <div class="gallery">
          {#each goods as g}
            <div class="g-item" title="{g.name}（形状 {g.shape} · 颜色 {g.color}）">
              <img src={imgUrl(g.id)} alt={g.name} onerror={(e) => (e.currentTarget.style.opacity = 0.15)} />
              <span class="g-name">{g.name}</span>
            </div>
          {/each}
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h2>🎫 任务</h2>
          <button class="btn ghost sm" onclick={refreshTasks}>刷新</button>
        </div>
        {#if currentTask}
          <div class="task-current">
            <div class="task-id">{currentTask.id} <span class="task-state">已领取</span></div>
            <ul class="task-items">
              {#each currentTask.items || [] as it}
                <li><b>{it.color}{it.shape}</b> ×{it.count}</li>
              {/each}
            </ul>
            <div class="task-actions">
              <button class="btn accent sm" onclick={() => finishTask('complete')}>完成任务</button>
              <button class="btn ghost sm" onclick={() => finishTask('cancel')}>放弃</button>
            </div>
          </div>
        {:else}
          <div class="task-queue">
            {#each tasks.filter((t) => t.status === 'available') as t}
              <div class="task-row">
                <span class="task-row-id">{t.id}</span>
                <span class="task-row-info">{t.items?.length || 0} 种 / {t.total} 件</span>
                <button class="btn sm" onclick={() => claim(t.id)}>认领</button>
              </div>
            {:else}
              <div class="empty small"><p>{deploy.gateway_reachable ? '暂无待领取任务' : '无数据（未连接实际服务）'}</p></div>
            {/each}
          </div>
        {/if}
      </div>
    </section>
  </div>

  <!-- ============ 投放顺序表 ============ -->
  <section class="card table-card">
    <div class="card-head">
      <h2>📋 投放顺序表</h2>
      <span class="hint">序号 · 货物名称 · 图片 · 分拣成功总数量 · 储物盒（最新在顶部）</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th style="width:80px">序号</th>
            <th style="width:190px">货物名称</th>
            <th style="width:100px">图片</th>
            <th style="width:150px">分拣成功总数量</th>
            <th style="width:110px">储物盒</th>
            <th style="width:150px">表面信息</th>
            <th style="width:100px">计时</th>
          </tr>
        </thead>
        <tbody>
          {#each tableRows as r}
            <tr class={display.current && r.seq === display.current.seq ? 'current' : ''}>
              <td class="c-seq">{r.seq}</td>
              <td class="c-name">{r.name}</td>
              <td>
                {#if r.class}
                  <img class="c-img" src={imgUrl(r.class)} alt={r.name} onerror={(e) => (e.currentTarget.style.visibility = 'hidden')} />
                {:else}<span class="c-img-ph">—</span>{/if}
              </td>
              <td class="c-total">{r.cumulative}</td>
              <td class="c-bin">{r.bin} 号盒</td>
              <td class="c-marks">{(r.marks || ['无']).join('·')}{#if r.qr} · 二维码{/if}{#if r.text} · 文字{/if}</td>
              <td class="c-ts">{fmtTime((r.elapsed_ms || 0) / 1000)}</td>
            </tr>
          {:else}
            <tr><td colspan="7" class="c-empty">暂无投放记录 —— 分拣第一件货物后此处按顺序显示</td></tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- ============ 扫码弹窗 ============ -->
  {#if qrOpen}
    <div class="mask">
      <button class="mask-backdrop" aria-label="关闭扫码弹窗" onclick={closeQr}></button>
      <div class="modal">
        <button class="modal-close" onclick={closeQr}>✕</button>
        <h3>⌘ 扫描二维码 · 领取分拣任务</h3>
        {#if !qrRaw}
          <div class="scan-box">
            <video bind:this={qrVideoEl} autoplay playsinline muted></video>
            <div class="scan-frame"><span class="scan-line"></span></div>
            <p class="scan-hint">将现场任务二维码对准取景框，自动识别</p>
            {#if qrError}<p class="scan-err">{qrError}</p>{/if}
          </div>
          <div class="qr-extra">
            <input placeholder="或手动输入任务码（T-260901）" bind:value={qrCode} onkeydown={(e) => e.key === 'Enter' && claim()} />
            <button class="btn ghost" onclick={toggleDemoQr}>{showDemoQr ? '收起' : '演示二维码'}</button>
          </div>
          {#if showDemoQr}
            <div class="demo-grid">
              {#each demoList as t}
                <div class="demo-item"><img src={demoQrUrl(t.id)} alt={t.id} /><span>{t.id}</span></div>
              {/each}
            </div>
          {/if}
        {:else}
          <div class="scan-ok">
            <div class="ok-title">✓ 已识别任务二维码</div>
            <p class="ok-code">{qrRaw}</p>
            <label class="worker-row"><span>领取人</span><input bind:value={worker} /></label>
            <div class="task-actions center">
              <button class="btn accent" onclick={() => claim()}>🎫 确认领取（复位显示屏与储物盒）</button>
              <button class="btn ghost" onclick={() => { qrRaw = ''; startQrScan(); }}>重新扫描</button>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  {#if toast}<div class="toast {toast.kind}">{toast.text}</div>{/if}
</main>

<style>
  :global(*) { margin: 0; padding: 0; box-sizing: border-box; }
  :global(body) {
    font-family: 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;
    background: #ffffff; color: #1f2937; min-height: 100vh;
  }
  .shell { max-width: 1880px; margin: 0 auto; padding: 12px 16px 26px; }

  .topbar {
    display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 12px 18px; border-radius: 14px; margin-bottom: 10px;
    background: #fff; border: 1px solid #e2e8f0; box-shadow: 0 2px 12px rgba(15,23,42,.06);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-mark { font-size: 1.5rem; color: #0891b2; }
  .brand h1 { font-size: 1.15rem; color: #0f172a; letter-spacing: .04em; }
  .brand p { font-size: .72rem; color: #64748b; margin-top: 3px; }
  .statusbar { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; font-size: .72rem;
    padding: 4px 10px; border-radius: 12px; background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569;
  }
  .pill b { color: #0369a1; font-size: .82rem; }
  .pill.timer { font-family: 'SF Mono', Consolas, monospace; font-weight: 700; color: #0f172a; font-size: .82rem; }
  .pill.clock { font-family: 'SF Mono', Consolas, monospace; color: #0e7490; }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot.ok { background: #16a34a; } .dot.err { background: #dc2626; }
  .dot.idle { background: #ca8a04; } .dot.warn { background: #f59e0b; animation: blink 1s infinite; }
  @keyframes blink { 0%,100% { opacity: 1 } 50% { opacity: .35 } }

  .banner {
    display: flex; align-items: center; gap: 10px; padding: 10px 18px; border-radius: 12px;
    margin-bottom: 10px; font-size: .92rem; font-weight: 600; border: 1px solid; flex-wrap: wrap;
  }
  .banner.running { background: #eff6ff; border-color: #93c5fd; color: #1e40af; }
  .banner.finished { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
  .banner.ready { background: #f0fdf4; border-color: #86efac; color: #166534; }
  .banner.offline { background: #fffbeb; border-color: #fcd34d; color: #92400e; }
  .banner.offline code { background: rgba(146, 64, 14, .12); padding: 1px 5px; border-radius: 5px; font-size: .74rem; }
  .banner-icon { font-size: 1.15rem; }
  .banner-text { flex: 1; }

  .layout { display: grid; grid-template-columns: 1fr 430px 300px; gap: 12px; align-items: start; }
  @media (max-width: 1560px) { .layout { grid-template-columns: 1fr 400px; } .col.side { grid-column: 1 / -1; } .side .gallery { grid-template-columns: repeat(8, 1fr); } }
  @media (max-width: 1080px) { .layout { grid-template-columns: 1fr; } }
  .col { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px; box-shadow: 0 1px 8px rgba(15,23,42,.05); }
  .card-head { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
  .card-head h2 { font-size: .95rem; color: #0f172a; }
  .hint { margin-left: auto; font-size: .68rem; color: #94a3b8; }

  .btn {
    border: 1px solid #cbd5e1; background: #f8fafc; color: #334155; border-radius: 10px;
    padding: 8px 16px; font-size: .82rem; cursor: pointer; transition: all .18s; white-space: nowrap;
  }
  .btn:hover:not(:disabled) { background: #e2e8f0; transform: translateY(-1px); }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn.primary { background: linear-gradient(135deg, #0891b2, #06b6d4); border-color: transparent; color: #fff; font-weight: 600; }
  .btn.accent { background: linear-gradient(135deg, #059669, #10b981); border-color: transparent; color: #fff; font-weight: 600; }
  .btn.danger { background: #fee2e2; border-color: #fca5a5; color: #b91c1c; font-weight: 600; }
  .btn.ghost { background: #fff; }
  .btn.sm { padding: 5px 12px; font-size: .74rem; }

  .cam-tools { margin-left: auto; display: flex; gap: 6px; }
  .src-switch { display: flex; gap: 4px; margin-left: 8px; }
  .src-btn {
    display: inline-flex; align-items: center; gap: 5px; font-size: .7rem; cursor: pointer;
    padding: 5px 10px; border-radius: 9px; border: 1px solid #e2e8f0; background: #f8fafc; color: #64748b;
  }
  .src-btn.active { background: #0891b2; border-color: #0891b2; color: #fff; font-weight: 700; }
  .src-dot { width: 6px; height: 6px; border-radius: 50%; background: #94a3b8; }
  .src-dot.ok { background: #22c55e; } .src-dot.err { background: #ef4444; }
  .cam-label.off { color: #b91c1c; }
  .warn-text { color: #b45309; }

  /* 自动分拣 */
  .auto-card { border-color: #bbf7d0; }
  .auto-body { display: flex; flex-direction: column; gap: 10px; }
  .auto-state {
    display: inline-flex; align-items: center; gap: 8px; align-self: flex-start;
    font-size: .8rem; font-weight: 700; padding: 5px 12px; border-radius: 10px;
    background: #f1f5f9; color: #475569;
  }
  .auto-state.run { background: #dcfce7; color: #15803d; }
  .auto-state.err { background: #fee2e2; color: #b91c1c; }
  .auto-state.done { background: #e0f2fe; color: #0369a1; }
  .auto-state-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: blink 1.4s infinite; }
  .auto-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .metric { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; }
  .metric-num { font-size: 1.1rem; font-weight: 800; color: #0f172a; }
  .metric-num.small { font-size: .82rem; }
  .metric-label { font-size: .64rem; color: #94a3b8; }
  .auto-err { font-size: .72rem; color: #b45309; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 6px 10px; }
  .auto-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .mini-field { display: inline-flex; align-items: center; gap: 6px; font-size: .72rem; color: #64748b; }
  .mini-field input { width: 74px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 8px; font-size: .76rem; }
  .chk { display: inline-flex; align-items: center; gap: 5px; font-size: .72rem; color: #64748b; cursor: pointer; }
  .auto-flow { font-size: .66rem; color: #94a3b8; line-height: 1.6; border-top: 1px dashed #e2e8f0; padding-top: 8px; }
  .tag.box { font-family: 'SF Mono', Consolas, monospace; font-size: .62rem; }
  .mjpeg-input { width: 210px; background: #fff; border: 1px solid #cbd5e1; color: #0f172a; border-radius: 8px; padding: 6px 10px; font-size: .72rem; }
  .cam-view { position: relative; aspect-ratio: 16/9; background: #0f172a; border-radius: 10px; overflow: hidden; border: 1px solid #cbd5e1; }
  .cam-view video, .cam-view > img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .cam-view .snapshot { position: absolute; inset: 0; }
  .cam-view .overlay { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
  .cam-holder { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; color: #94a3b8; }
  .cam-icon { font-size: 2.2rem; opacity: .6; }
  .cam-holder .sub { font-size: .7rem; }
  .cam-label { position: absolute; top: 10px; left: 10px; font-size: .66rem; padding: 3px 10px; border-radius: 10px; background: rgba(255,255,255,.9); color: #dc2626; font-weight: 700; }
  .cam-actions { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .cam-actions .hint { margin-left: 0; }

  .ai-result { margin-top: 12px; border-top: 1px dashed #e2e8f0; padding-top: 10px; }
  .ai-head { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
  .ai-badge { font-size: .7rem; background: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 10px; font-weight: 600; }
  .ai-note { font-size: .68rem; color: #94a3b8; }
  .ai-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid #f1f5f9; flex-wrap: wrap; }
  .ai-idx { font-size: .7rem; color: #94a3b8; width: 26px; }
  .ai-name { font-size: .9rem; font-weight: 700; color: #0f172a; min-width: 90px; }
  .tag { font-size: .68rem; background: #f1f5f9; color: #475569; padding: 3px 9px; border-radius: 9px; }
  .tag b { color: #0369a1; }
  .tag.conf { background: #dcfce7; color: #15803d; font-weight: 600; }
  .ai-qr { margin-top: 8px; font-size: .78rem; color: #0f172a; background: #f8fafc; padding: 8px 10px; border-radius: 9px; }

  .debug-toggle { margin-left: auto; display: inline-flex; align-items: center; gap: 6px; font-size: .72rem; color: #64748b; cursor: pointer; }
  .quick-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  @media (max-width: 760px) { .quick-grid { grid-template-columns: 1fr 1fr; } }
  .qbtn {
    display: flex; flex-direction: column; gap: 3px; text-align: left; min-height: 82px;
    padding: 12px 14px; border-radius: 12px; cursor: pointer; font-family: inherit;
    background: #f8fafc; border: 1px solid #e2e8f0; transition: all .18s;
  }
  .qbtn:hover:not(:disabled) { border-color: var(--c); box-shadow: 0 3px 14px rgba(15,23,42,.1); transform: translateY(-2px); }
  .qbtn:disabled { opacity: .45; cursor: not-allowed; }
  .qicon { font-size: 1.1rem; }
  .qtext { font-size: .86rem; font-weight: 700; color: #0f172a; }
  .qdesc { font-size: .66rem; color: #94a3b8; }
  .qbtn.danger .qtext { color: #b91c1c; }

  .manual-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e2e8f0; flex-wrap: wrap; }
  .manual-label { font-size: .78rem; font-weight: 700; color: #0f172a; }
  .manual-row select, .mini-input {
    border: 1px solid #cbd5e1; border-radius: 9px; padding: 7px 10px; font-size: .78rem;
    background: #fff; color: #0f172a; font-family: inherit;
  }
  .mini-input { width: 110px; }
  .manual-tip { font-size: .66rem; color: #b45309; }

  .current-card { border-color: #bae6fd; }
  .cur-grid { display: grid; grid-template-columns: 92px 128px 1fr; gap: 12px; align-items: center; }
  .cur-seq { text-align: center; background: #eff6ff; border-radius: 12px; padding: 10px 6px; }
  .cur-seq-label { display: block; font-size: .66rem; color: #64748b; }
  .cur-seq-num { font-size: 2rem; font-weight: 800; color: #1d4ed8; line-height: 1.1; }
  .cur-img-box { position: relative; width: 128px; height: 128px; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; background: #fff; }
  .cur-img-box img { width: 100%; height: 100%; object-fit: contain; }
  .cur-img-ph { display: flex; align-items: center; justify-content: center; height: 100%; color: #cbd5e1; font-size: .74rem; }
  .fmt-badge { position: absolute; right: 6px; bottom: 6px; font-size: .58rem; background: rgba(15,23,42,.75); color: #fff; padding: 2px 6px; border-radius: 6px; }
  .cur-info { min-width: 0; }
  .cur-name { font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; }
  .cur-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .cur-extra { font-size: .72rem; color: #475569; margin-bottom: 4px; }
  .cur-total { display: flex; align-items: baseline; gap: 8px; }
  .cur-total-label { font-size: .72rem; color: #64748b; }
  .cur-total-num { font-size: 2.1rem; font-weight: 800; color: #059669; line-height: 1; }
  .cur-ts { font-size: .66rem; color: #94a3b8; margin-top: 6px; }
  .cur-state { margin-top: 10px; font-size: .78rem; font-weight: 600; padding: 6px 12px; border-radius: 9px; }
  .cur-state.locked { background: #fffbeb; color: #92400e; }
  .cur-state.ready { background: #f0fdf4; color: #166534; }
  .empty { text-align: center; padding: 22px 10px; color: #94a3b8; }
  .empty-icon { font-size: 2rem; opacity: .5; margin-bottom: 6px; }
  .empty .sub { font-size: .68rem; margin-top: 6px; line-height: 1.6; }
  .empty.small { padding: 10px; }

  /* 储物盒 */
  .bin-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .bin { border: 1px solid #e2e8f0; border-radius: 12px; padding: 8px; background: #f8fafc; min-height: 104px; display: flex; flex-direction: column; gap: 4px; }
  .bin.empty { background: #fff; border-style: dashed; }
  .bin.used { border-color: #7dd3fc; background: #f0f9ff; }
  .bin.full { border-color: #fcd34d; background: #fffbeb; }
  .bin-no { font-size: 1.1rem; font-weight: 800; color: #0f172a; }
  .bin-unit { font-size: .62rem; color: #94a3b8; margin-left: 2px; }
  .bin-key { font-size: .72rem; color: #475569; font-weight: 600; }
  .bin.empty .bin-key { color: #cbd5e1; font-weight: 400; }
  .bin-thumbs { display: flex; gap: 3px; flex-wrap: wrap; }
  .bin-thumbs img { width: 26px; height: 26px; object-fit: contain; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }
  .bin-bar { height: 5px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
  .bin-bar span { display: block; height: 100%; background: #0891b2; }
  .bin.full .bin-bar span { background: #f59e0b; }
  .bin-count { font-size: .68rem; color: #64748b; }
  .bin-count.full { color: #b45309; font-weight: 700; }
  .bin-rule { margin-top: 10px; font-size: .68rem; color: #94a3b8; line-height: 1.6; }

  .fmt-switch { margin-left: auto; display: flex; gap: 4px; flex-wrap: wrap; }
  .fmt-btn {
    font-size: .62rem; padding: 3px 7px; border-radius: 8px; cursor: pointer;
    border: 1px solid #e2e8f0; background: #f8fafc; color: #64748b; text-transform: uppercase;
  }
  .fmt-btn.active { background: #0891b2; border-color: #0891b2; color: #fff; font-weight: 700; }
  .gallery { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .g-item { text-align: center; border: 1px solid #e2e8f0; border-radius: 10px; padding: 6px 4px; background: #fff; }
  .g-item img { width: 100%; height: 52px; object-fit: contain; }
  .g-name { display: block; font-size: .6rem; color: #334155; margin-top: 3px; }

  .task-current { border: 1px solid #bae6fd; background: #f0f9ff; border-radius: 12px; padding: 10px; }
  .task-id { font-size: .95rem; font-weight: 700; color: #0369a1; display: flex; gap: 8px; align-items: center; }
  .task-state { font-size: .6rem; background: #dcfce7; color: #15803d; padding: 2px 7px; border-radius: 8px; }
  .task-items { list-style: none; margin: 8px 0; display: flex; flex-direction: column; gap: 4px; font-size: .76rem; color: #334155; }
  .task-actions { display: flex; gap: 8px; margin-top: 8px; }
  .task-actions.center { justify-content: center; margin-top: 14px; }
  .task-queue { display: flex; flex-direction: column; gap: 6px; max-height: 240px; overflow-y: auto; }
  .task-row { display: flex; align-items: center; gap: 8px; padding: 7px 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; }
  .task-row-id { font-size: .78rem; font-weight: 700; color: #0f172a; }
  .task-row-info { font-size: .66rem; color: #94a3b8; flex: 1; }

  .table-card { margin-top: 12px; }
  .table-wrap { max-height: 300px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: .84rem; }
  thead th {
    position: sticky; top: 0; background: #f8fafc; color: #475569; font-size: .74rem;
    text-align: left; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; z-index: 1;
  }
  tbody td { padding: 8px 14px; border-bottom: 1px solid #f1f5f9; color: #334155; }
  tr.current { background: #f0f9ff; }
  tr.current td { font-weight: 600; }
  .c-seq { font-size: 1.05rem; font-weight: 800; color: #1d4ed8; }
  .c-name { font-weight: 700; color: #0f172a; }
  .c-img { width: 42px; height: 42px; object-fit: contain; vertical-align: middle; }
  .c-img-ph { color: #cbd5e1; }
  .c-total { font-size: 1.15rem; font-weight: 800; color: #059669; }
  .c-bin { color: #0369a1; font-weight: 700; }
  .c-marks { font-size: .74rem; color: #64748b; }
  .c-ts { color: #94a3b8; font-size: .74rem; font-family: 'SF Mono', Consolas, monospace; }
  .c-empty { text-align: center; color: #94a3b8; padding: 24px; }

  .mask { position: fixed; inset: 0; z-index: 100; background: rgba(15,23,42,.55); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; padding: 16px; }
  .mask-backdrop { position: absolute; inset: 0; width: 100%; height: 100%; border: none; background: transparent; cursor: default; padding: 0; }
  .modal { position: relative; z-index: 1; width: 100%; max-width: 500px; background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; box-shadow: 0 20px 50px rgba(15,23,42,.25); }
  .modal h3 { font-size: 1rem; color: #0f172a; margin-bottom: 12px; }
  .modal-close { position: absolute; top: 12px; right: 14px; width: 30px; height: 30px; border-radius: 50%; border: 1px solid #cbd5e1; background: #f8fafc; color: #475569; cursor: pointer; }
  .scan-box { position: relative; aspect-ratio: 4/3; border-radius: 12px; overflow: hidden; background: #0f172a; }
  .scan-box video { width: 100%; height: 100%; object-fit: cover; }
  .scan-frame::before { content: ''; position: absolute; top: 12%; left: 12%; right: 12%; bottom: 12%; border: 2px solid rgba(8,145,178,.85); border-radius: 14px; }
  .scan-line { position: absolute; left: 14%; right: 14%; height: 2px; background: linear-gradient(90deg, transparent, #06b6d4, transparent); animation: scan 2s ease-in-out infinite; }
  @keyframes scan { 0%,100% { top: 16% } 50% { top: 82% } }
  .scan-hint { text-align: center; font-size: .72rem; color: #64748b; margin-top: 10px; }
  .scan-err { text-align: center; font-size: .72rem; color: #b91c1c; margin-top: 6px; }
  .qr-extra { display: flex; gap: 8px; margin-top: 12px; }
  .qr-extra input, .worker-row input { flex: 1; border: 1px solid #cbd5e1; border-radius: 9px; padding: 9px 12px; font-size: .8rem; color: #0f172a; background: #fff; }
  .demo-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
  .demo-item { text-align: center; font-size: .62rem; color: #64748b; }
  .demo-item img { width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; display: block; }
  .scan-ok { text-align: center; }
  .ok-title { font-size: .95rem; font-weight: 700; color: #059669; margin-bottom: 8px; }
  .ok-code { font-family: 'SF Mono', Consolas, monospace; font-size: .78rem; background: #f1f5f9; padding: 8px; border-radius: 8px; word-break: break-all; margin-bottom: 12px; color: #0f172a; }
  .worker-row { display: flex; align-items: center; gap: 8px; font-size: .78rem; color: #64748b; }

  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 200;
    background: #fff; border: 1px solid #cbd5e1; color: #0f172a; padding: 10px 22px;
    border-radius: 12px; font-size: .84rem; box-shadow: 0 10px 30px rgba(15,23,42,.18);
  }
  .toast.err { border-color: #fecaca; color: #b91c1c; }
  .toast.warn { border-color: #fde68a; color: #92400e; }
</style>

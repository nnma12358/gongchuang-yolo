#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_app.py —— 浏览器里的标注复核工具（零依赖，只用标准库）
=====================================================================
为什么自己做：复核 150 张图的瓶颈不是"能不能画框"，而是**每张图的决策次数**。
X-AnyLabeling 这类工具要点选、拖拽、按保存；这里把 90% 的情况压成**一次按键**：
现有框基本对 → 按 A 下一张；框偏了 → 拖一下/微调；漏了 → 按 N 采纳模型建议框。

启动：
    python3 tools/review_app.py                       # 默认 0.0.0.0:8770
    python3 tools/review_app.py --dir data/review/review_set --port 8770
然后浏览器打开（Windows 侧同样可访问）：  http://localhost:8770

键盘（鼠标拖拽 = 在当前标注上继续画框/改框）：
    鼠标拖拽     空白处拖 = 新建框；框内拖 = 移动；**拖边框/角上的白色手柄 = 调整大小**
    A / Enter  标注正确，下一张（记为"已复核"）
    N          并入模型建议框（自动去重，不会越按越多；Shift+N 强制追加）
    Z          撤销本张所有修改    D / Del 删除选中的框    1..9 选中第 N 个框
    方向键      微调选中框位置（Shift=大步长，Alt=改尺寸）；没选中框时 ←/→ 翻页
    , / .      上一张 / 下一张     W 跳到下一个未复核     G 跳到指定序号
    Esc        取消选中            H 帮助
自动保存：任何修改立即写入 labels/*.txt，刷新/断电都不丢。
"""
import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>标注复核</title>
<style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;background:#11161d;color:#e8eef6;
   font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
 #wrap{display:flex;height:100%}
 #left{flex:0 0 260px;border-right:1px solid #253040;display:flex;flex-direction:column;min-width:0}
 #right{flex:1;display:flex;flex-direction:column;min-width:0}
 .hdr{padding:10px 12px;border-bottom:1px solid #253040;background:#161d27}
 #list{flex:1;overflow:auto}
 .it{padding:7px 12px;cursor:pointer;border-bottom:1px solid #1c2532;display:flex;gap:8px;align-items:center}
 .it:hover{background:#1c2532}
 .it.cur{background:#1d3a5c}
 .it .sc{flex:0 0 34px;text-align:right;font-variant-numeric:tabular-nums;color:#ffb454}
 .it .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12.5px}
 .it .dot{flex:0 0 8px;height:8px;border-radius:50%;background:#3a4756}
 .it.done .dot{background:#3ddc84}
 .it.done .nm{color:#8fa3b8}
 #bar{padding:8px 12px;border-bottom:1px solid #253040;display:flex;gap:14px;align-items:center;background:#161d27}
 #bar b{color:#7fd1ff}
 #stage{flex:1;position:relative;overflow:hidden;background:#0b0f14;display:flex;align-items:center;justify-content:center;min-height:0}
 canvas{cursor:crosshair;image-rendering:auto;flex:none}
 #info{padding:8px 12px;border-top:1px solid #253040;background:#161d27;font-size:13px;min-height:42px}
 .tag{display:inline-block;padding:1px 7px;border-radius:10px;margin-right:6px;font-size:12px}
 .t1{background:#25402a;color:#8ff0a4}.t2{background:#402a25;color:#ffb3a0}.t3{background:#2a3040;color:#a9c4ff}
 .t4{background:#403a25;color:#ffdd9e}
 kbd{background:#222c39;border:1px solid #35455a;border-radius:4px;padding:0 5px;font-size:12px}
 .pill{background:#222c39;border-radius:6px;padding:2px 8px}
</style></head><body>
<div id="wrap">
  <div id="left">
    <div class="hdr"><b>待复核 <span id="cnt"></span></b><div style="color:#8fa3b8;font-size:12px" id="prog"></div></div>
    <div id="list"></div>
  </div>
  <div id="right">
    <div id="bar">
      <span class="pill" id="pos">-/-</span>
      <span id="name"></span>
      <span style="margin-left:auto"><b>A</b> 正确·下一张 &nbsp;<b>N</b> 并入模型框 &nbsp;<b>拖拽</b> 画框/拖边框调大小 &nbsp;<b>Z</b> 撤销 &nbsp;<b>, .</b> 翻页 &nbsp;<b>W</b> 下一张未复核 &nbsp;<b>H</b> 帮助</span>
    </div>
    <div id="stage"><canvas id="cv"></canvas></div>
    <div id="info"></div>
  </div>
</div>
<script>
const S={items:[],i:0,boxes:[],hints:[],orig:[],sel:-1,img:null,drag:null,meta:{},done:{},hover:-1};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');

async function boot(){
  const d=await (await fetch('/api/list')).json();
  S.items=d.items; S.meta=d.meta||{}; S.done=d.done||{};
  renderList(); goto(0);
}
function renderList(){
  const el=document.getElementById('list'); el.innerHTML='';
  let doneN=0;
  S.items.forEach((it,k)=>{
    if(S.done[it.file]) doneN++;
    const d=document.createElement('div');
    d.className='it'+(k===S.i?' cur':'')+(S.done[it.file]?' done':'');
    d.innerHTML=`<span class="sc">${it.score}</span><span class="dot"></span>
      <span class="nm">${it.file.replace(/^\d+_/,'')}</span>`;
    d.onclick=()=>goto(k);
    el.appendChild(d);
  });
  document.getElementById('cnt').textContent=S.items.length+' 张';
  document.getElementById('prog').textContent='已复核 '+doneN+' / '+S.items.length;
}
function fit(){
  const st=document.getElementById('stage');
  const maxW=st.clientWidth-16, maxH=st.clientHeight-16;
  if(!S.img||!S.img.width) return;
  const s=Math.min(maxW/S.img.width, maxH/S.img.height);
  cv.width=Math.max(64,Math.round(S.img.width*s));
  cv.height=Math.max(64,Math.round(S.img.height*s));
  cv.style.width=cv.width+'px'; cv.style.height=cv.height+'px';
}
function draw(){
  if(!S.img||!S.img.width){                 // 图片没加载出来就别画，避免把画布搞坏
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.fillStyle='#8899aa'; ctx.font='16px sans-serif';
    ctx.fillText('图片未加载：请刷新页面（或检查 /img/ 是否 404）',16,32);
    return;
  }
  // ⚠ 关键：框坐标是**归一化**的(0~1)，必须乘画布尺寸 W/H；
  //   之前乘的是 s=cv.width/img.width（像素缩放比），框全被画到左上角 0.x 像素处，
  //   看起来就是"什么都看不到"。
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);
  ctx.drawImage(S.img,0,0,W,H);
  // 模型建议框（虚线）
  S.hints.forEach(h=>{
    const x1=h.xyxy[0]*W, y1=h.xyxy[1]*H, x2=h.xyxy[2]*W, y2=h.xyxy[3]*H;
    ctx.setLineDash([7,4]); ctx.lineWidth=2;
    ctx.strokeStyle=h.conf>=0.25?'rgba(255,90,90,.95)':'rgba(255,170,60,.9)';
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.setLineDash([]);
    ctx.fillStyle=ctx.strokeStyle; ctx.font='bold 13px sans-serif';
    ctx.fillText('M'+(h.conf*100|0)/100,x1+3,Math.max(14,y1-4));
  });
  // 当前标注
  S.boxes.forEach((b,k)=>{
    const on=k===S.sel;
    ctx.lineWidth=on?4:3;
    ctx.strokeStyle=on?'#ffd166':'#22d46e';
    ctx.strokeRect(b[0]*W,b[1]*H,b[2]*W,b[3]*H);
    ctx.fillStyle=on?'#ffd166':'#22d46e'; ctx.font='bold 14px sans-serif';
    ctx.fillText(String(k+1),b[0]*W+4,Math.min(H-4,b[1]*H+16));
    if(on){                       // 选中框画 8 个手柄，提示"可以拖边框调大小"
      const x1=b[0]*W, y1=b[1]*H, x2=x1+b[2]*W, y2=y1+b[3]*H;
      ctx.fillStyle='#ffffff'; ctx.strokeStyle='#ffd166'; ctx.lineWidth=2;
      [[x1,y1],[(x1+x2)/2,y1],[x2,y1],[x2,(y1+y2)/2],[x2,y2],[(x1+x2)/2,y2],[x1,y2],[x1,(y1+y2)/2]]
        .forEach(([hx,hy])=>{ ctx.fillRect(hx-4,hy-4,8,8); ctx.strokeRect(hx-4,hy-4,8,8); });
    }
  });
}
async function goto(k){
  if(k<0||k>=S.items.length) return;
  S.i=k; S.sel=-1; S.drag=null;
  const it=S.items[k];
  const d=await (await fetch('/api/item?file='+encodeURIComponent(it.file))).json();
  S.boxes=d.boxes.map(b=>b.slice()); S.orig=d.boxes.map(b=>b.slice()); S.hints=d.hints||[];
  S.img=new Image();
  S.img.onload=()=>{fit();draw();};
  S.img.onerror=()=>{ showErr('图片加载失败：'+it.file); };
  S.img.src='/img/'+encodeURIComponent(it.file);
  document.getElementById('name').textContent=it.file;
  document.getElementById('pos').textContent=(k+1)+'/'+S.items.length;
  const w=it.why?(' <span class="tag t2">'+it.why+'</span>'):'';
  document.getElementById('info').innerHTML=
    `<span class="tag t1">${it.kind==='unlabeled'?'待从零标注':'已有标注'}</span>`+
    `<span class="tag t3">${it.source}</span>`+
    `<span class="tag t4">v3:${it.v3_split}</span>`+w+
    `<span style="color:#8fa3b8"> 当前 ${S.boxes.length} 框`
    +(S.hints.length?` · 模型建议 ${S.hints.length} 个（按 N 采纳）`:``)+`</span>`;
  renderList();
}
function norm(ev){
  const r=cv.getBoundingClientRect();
  const x=(ev.clientX-r.left)/cv.width, y=(ev.clientY-r.top)/cv.height;
  return [Math.min(1,Math.max(0,x)),Math.min(1,Math.max(0,y))];
}
function hit(x,y){
  for(let k=S.boxes.length-1;k>=0;k--){const b=S.boxes[k];
    if(x>=b[0]&&x<=b[0]+b[2]&&y>=b[1]&&y<=b[1]+b[3]) return k;}
  return -1;
}
function edgeHit(x,y){        // 命中框的边/角 → 调整大小（阈值按画布像素算，手感稳定）
  const tolX=8/Math.max(1,cv.width), tolY=8/Math.max(1,cv.height);
  for(let k=S.boxes.length-1;k>=0;k--){
    const b=S.boxes[k], x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3];
    if(x<x1-tolX||x>x2+tolX||y<y1-tolY||y>y2+tolY) continue;
    const l=Math.abs(x-x1)<=tolX, r=Math.abs(x-x2)<=tolX;
    const t=Math.abs(y-y1)<=tolY, bo=Math.abs(y-y2)<=tolY;
    if(l||r||t||bo) return {k:k, l:l, r:r, t:t, b:bo,
                            x1:x1, y1:y1, x2:x2, y2:y2, cx:x, cy:y};
  }
  return null;
}
cv.onmousedown=e=>{
  const [x,y]=norm(e);
  const eh=edgeHit(x,y);
  if(eh){ S.sel=eh.k; S.drag={mode:'resize',eh:eh}; }
  else { const k=hit(x,y);
    if(k>=0){ S.sel=k; S.drag={mode:'move',x,y,box:S.boxes[k].slice()}; }
    else { S.boxes.push([x,y,0,0]); S.sel=S.boxes.length-1;
           S.drag={mode:'new',x,y,box:S.boxes[S.sel]}; } }
  draw();
};
cv.onmousemove=e=>{
  const [x,y]=norm(e);
  if(!S.drag){
    const eh=edgeHit(x,y);
    const k=eh?eh.k:hit(x,y);
    if(k!==S.hover||!!eh!==!!S.hoverEdge){ S.hover=k; S.hoverEdge=!!eh; }
    let cur='crosshair';
    if(eh){
      if((eh.l&&eh.t)||(eh.r&&eh.b)) cur='nwse-resize';
      else if((eh.r&&eh.t)||(eh.l&&eh.b)) cur='nesw-resize';
      else if(eh.l||eh.r) cur='ew-resize';
      else cur='ns-resize';
    } else if(k>=0) cur='move';
    cv.style.cursor=cur;
    return;
  }
  const d=S.drag;
  if(d.mode==='new'){
    S.boxes[S.sel]=[Math.min(d.x,x),Math.min(d.y,y),Math.abs(x-d.x),Math.abs(y-d.y)];
  } else if(d.mode==='move'){
    const o=d.box;
    S.boxes[S.sel]=[o[0]+(x-d.x),o[1]+(y-d.y),o[2],o[3]];
  } else {                                   // resize：被拖的边跟着走，对边固定
    const eh=d.eh;
    let x1=eh.l?x:eh.x1, x2=eh.r?x:eh.x2, y1=eh.t?y:eh.y1, y2=eh.b?y:eh.y2;
    if(x2<x1){ const s=x1; x1=x2; x2=s; }
    if(y2<y1){ const s=y1; y1=y2; y2=s; }
    S.boxes[S.sel]=[x1,y1,x2-x1,y2-y1];
  }
  draw();
};
cv.onmouseup=()=>{ if(!S.drag) return;
  const b=S.boxes[S.sel];
  if(b[2]<0.004&&b[3]<0.004&&S.drag.mode==='new'){S.boxes.splice(S.sel,1);S.sel=-1;}
  else S.boxes[S.sel]=clamp(b);
  S.drag=null; draw(); save(); };
function clamp(b){
  let [x,y,w,h]=b; w=Math.max(0.004,Math.min(1,w)); h=Math.max(0.004,Math.min(1,h));
  x=Math.max(0,Math.min(1-w,x)); y=Math.max(0,Math.min(1-h,y));
  return [x,y,w,h];
}
function showErr(msg){
  const el=document.getElementById('info');
  el.innerHTML='<span class="tag t2">'+msg+'</span>';
}
function near(a,b,thr){       // 两个框是否几乎重合（用于去重，避免反复按 N 堆叠）
  const ix=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]));
  const iy=Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));
  const inter=ix*iy, ua=a[2]*a[3]+b[2]*b[3]-inter;
  return ua>0 && inter/ua>=thr;
}
async function save(){
  const it=S.items[S.i];
  await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:it.file,boxes:S.boxes,reviewed:!!S.done[it.file]})});
  document.getElementById('info').innerHTML=
    document.getElementById('info').innerHTML.replace(/当前 \d+ 框/,'当前 '+S.boxes.length+' 框');
}
async function markDone(v){
  const it=S.items[S.i]; S.done[it.file]=v;
  await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:it.file,boxes:S.boxes,reviewed:!!v})});
  renderList();
}
document.addEventListener('keydown',async e=>{
  const k=e.key;
  if(k==='a'||k==='A'||k==='Enter'){ e.preventDefault(); await markDone(true); next(1,true); return; }
  if(k==='w'||k==='W'){ e.preventDefault(); let j=S.i+1; while(j<S.items.length&&S.done[S.items[j].file]) j++;
    goto(j<S.items.length?j:0); return; }
  // ← → ：没选中框时翻页；选中框时用于微调（见下）
  if((k==='ArrowRight'||k==='ArrowLeft')&&S.sel<0){
    e.preventDefault(); await goto(S.i+(k==='ArrowRight'?1:-1)); return; }
  if(k==='.'||k==='>'){ e.preventDefault(); await goto(S.i+1); return; }
  if(k===','||k==='<'){ e.preventDefault(); await goto(S.i-1); return; }
  if(k==='z'||k==='Z'){ S.boxes=S.orig.map(b=>b.slice()); S.sel=-1; draw(); save(); return; }
  if(k==='n'||k==='N'){                       // 并入模型建议框（去重，不会越按越多）
    let add=0;
    S.hints.forEach(h=>{ const x1=h.xyxy[0],y1=h.xyxy[1];
      const nb=clamp([x1,y1,h.xyxy[2]-x1,h.xyxy[3]-y1]);
      if(!S.boxes.some(b=>near(b,nb,0.75))){ S.boxes.push(nb); add++; }
      else if(e.shiftKey){ S.boxes.push(nb); add++; }   // Shift+N 强制追加
    });
    draw(); await save();
    alert(add?('已并入 '+add+' 个模型建议框（共 '+S.boxes.length+' 框）')
             :'模型建议框已全部存在，没有新增（要重复添加请按 Shift+N）');
    return; }
  if(k==='d'||k==='D'||k==='Delete'||k==='Backspace'){
    if(S.sel>=0){ S.boxes.splice(S.sel,1); S.sel=-1; draw(); save(); } return; }
  if(/^[1-9]$/.test(k)){ const n=+k-1; if(n<S.boxes.length){S.sel=n;draw();} return; }
  if(k==='Escape'){ S.sel=-1; draw(); return; }
  if(k==='g'||k==='G'){ const v=prompt('跳到第几张（1-'+S.items.length+'）：'); 
    if(v){const n=parseInt(v,10); if(n>=1&&n<=S.items.length) goto(n-1);} return; }
  if(k==='h'||k==='H'){ alert(document.querySelector('#bar span:last-child').innerText
    .replace(/ /g,'\n')); return; }
  if(k.startsWith('Arrow')&&S.sel>=0){
    let step=e.shiftKey?0.02:0.003; const b=S.boxes[S.sel].slice();
    if(e.altKey){ if(k==='ArrowRight')b[2]+=step; if(k==='ArrowLeft')b[2]-=step;
                  if(k==='ArrowDown')b[3]+=step; if(k==='ArrowUp')b[3]-=step; }
    else { if(k==='ArrowRight')b[0]+=step; if(k==='ArrowLeft')b[0]-=step;
           if(k==='ArrowDown')b[1]+=step; if(k==='ArrowUp')b[1]-=step; }
    S.boxes[S.sel]=clamp(b); draw(); save(); e.preventDefault();
  }
});
async function next(d,skipUnreviewed){ await goto(S.i+d); }
window.addEventListener('resize',()=>{fit();draw();});
boot();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):        # 静音访问日志
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---------- GET ----------
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/list":
            return self._json({"items": self.server.items, "meta": self.server.meta,
                               "done": self.server.state})
        if u.path == "/api/item":
            f = os.path.basename(unquote(u.query.split("file=")[-1]))
            lp = os.path.join(self.server.dir, "labels", os.path.splitext(f)[0] + ".txt")
            boxes = []
            if os.path.exists(lp):
                for line in open(lp, encoding="utf-8"):
                    p = line.split()
                    if len(p) >= 5:
                        cx, cy, w, h = [float(v) for v in p[1:5]]
                        boxes.append([cx - w / 2, cy - h / 2, w, h])
            key = self.server.orig_of.get(f, f)
            return self._json({"boxes": boxes, "hints": self.server.hints.get(key, [])})
        if u.path.startswith("/img/"):
            f = os.path.basename(unquote(u.path[5:]))
            p = os.path.join(self.server.dir, "images", f)
            if not os.path.exists(p):
                return self._send(404, b"not found", "text/plain")
            with open(p, "rb") as fh:
                return self._send(200, fh.read(), "image/jpeg" if p.lower().endswith(
                    (".jpg", ".jpeg")) else "image/png")
        return self._send(404, b"not found", "text/plain")

    # ---------- POST ----------
    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/labels":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        f = os.path.basename(data.get("file", ""))
        boxes = data.get("boxes", [])
        lp = os.path.join(self.server.dir, "labels", os.path.splitext(f)[0] + ".txt")
        with self.server.lock:
            if boxes:
                with open(lp, "w", encoding="utf-8") as fh:
                    for x, y, w, h in boxes:
                        fh.write("0 %.6f %.6f %.6f %.6f\n" % (x + w / 2, y + h / 2, w, h))
            else:
                open(lp, "w").close()          # 空文件 = 该图无目标（负样本）
            self.server.state[f] = bool(data.get("reviewed"))
            json.dump(self.server.state,
                      open(os.path.join(self.server.dir, "review_state.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
        return self._json({"ok": True, "n": len(boxes)})


def build_items(rv_dir, audit_path):
    """复核清单：优先用 audit.json 的排序与理由（含模型建议框）"""
    items, hints, orig_of, meta = [], {}, {}, {}
    if os.path.exists(audit_path):
        a = json.load(open(audit_path, encoding="utf-8"))
        for m in a["mapping"]:
            items.append({"file": m["review_name"], "score": m["score"], "why": m.get("why", ""),
                          "kind": "labeled", "source": "", "v3_split": m.get("v3_split", "-")})
        by_orig = {r["image"]: r for r in a["items"]}
        for i, m in enumerate(a["mapping"]):
            r = by_orig.get(m["original"], {})
            items[i]["kind"] = r.get("kind", "labeled")
            items[i]["source"] = r.get("source", "")
            orig_of[m["review_name"]] = m["original"]
        h = os.path.join(os.path.dirname(audit_path), "hints.json")
        if os.path.exists(h):
            hints = json.load(open(h, encoding="utf-8"))
    else:   # 退化：直接扫目录
        for p in sorted(os.listdir(os.path.join(rv_dir, "images"))):
            items.append({"file": p, "score": 0, "why": "", "kind": "labeled",
                          "source": "", "v3_split": "-"})
    return items, hints, orig_of


def main():
    ap = argparse.ArgumentParser(description="标注复核 Web 工具")
    ap.add_argument("--dir", default="data/review/review_set")
    ap.add_argument("--audit", default="data/review/audit.json")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    rv = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    audit = args.audit if os.path.isabs(args.audit) else os.path.join(ROOT, args.audit)
    items, hints, orig_of = build_items(rv, audit)
    st_path = os.path.join(rv, "review_state.json")
    state = json.load(open(st_path, encoding="utf-8")) if os.path.exists(st_path) else {}

    srv = ThreadingHTTPServer((args.host, args.port), H)
    srv.items, srv.hints, srv.orig_of = items, hints, orig_of
    srv.state, srv.lock, srv.dir = state, threading.Lock(), rv
    srv.meta = {"dir": rv}
    done = sum(1 for k, v in state.items() if v)
    print("复核台已启动: http://localhost:%d   （%d 张，已复核 %d）" % (args.port, len(items), done))
    print("Ctrl+C 退出；标注实时保存到", os.path.join(rv, "labels"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()

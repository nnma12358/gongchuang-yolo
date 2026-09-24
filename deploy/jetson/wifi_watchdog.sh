#!/usr/bin/env bash
# ============================================================
# WiFi 看护（宿主机 systemd 常驻）
# ------------------------------------------------------------
# 现场故障现象：Nano 上 WiFi **显示已连接但没有 IP**（或走着走着丢 IP），
# 结果 SSH/ToDesk/网页全断。根因通常是三类，看护分别处理：
#
#   1) **已关联但拿不到 IPv4**（DHCP 失败/租约丢失）→ 重新拉连接（nmcli con up）
#   2) **完全没关联**（掉出热点）→ nmcli device connect wlan0
#   3) **网卡从 USB 上消失**（Realtek USB 网卡供电/驱动问题，或过流）
#      → 不刷屏、只记录；等它回来（拔插/复位后会自动重新连接）
#
# 另配：关闭 WiFi 省电（rtl88x2bu 开省电时非常容易掉线），见
#   /etc/NetworkManager/conf.d/99-sort-wifi-powersave-off.conf
#
# 用法：
#   bash wifi_watchdog.sh            # 前台
#   bash wifi_watchdog.sh --once     # 只查一轮
# 环境变量：IFACE=wlan0  USB_ID=0bda:b812  CHECK_INTERVAL=15  LOG=/tmp/wifi_watchdog.log
# ============================================================
set -uo pipefail

IFACE="${IFACE:-wlan0}"
USB_ID="${USB_ID:-0bda:b812}"
CHECK_INTERVAL="${CHECK_INTERVAL:-15}"
LOG="${LOG:-/tmp/wifi_watchdog.log}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

dongle_on_usb() { lsusb 2>/dev/null | grep -qi "$USB_ID"; }
iface_exists()  { ip link show "$IFACE" >/dev/null 2>&1; }
has_ipv4()      { ip -4 addr show "$IFACE" 2>/dev/null | grep -q "inet "; }
nm_state()      { nmcli -t -f GENERAL.STATE device show "$IFACE" 2>/dev/null | cut -d: -f2 | sed 's/ (.*//'; }
reconnect()     {
  log "  → 重新拉连接（nmcli device connect $IFACE）"
  nmcli device connect "$IFACE" >/dev/null 2>&1 || nmcli connection up "$IFACE" >/dev/null 2>&1
  sleep 5
  if has_ipv4; then
    log "  ✓ 已恢复：$(ip -4 -brief addr show "$IFACE" | awk '{print $3}')"
  else
    log "  ✗ 仍未拿到 IP（再等下一轮；若持续失败建议查热点/供电）"
  fi
}

check_once() {
  if ! dongle_on_usb; then
    log "⚠ WiFi 网卡不在 USB 上（$USB_ID）—— 多半是供电/驱动问题，等它回来"
    return 0
  fi
  if ! iface_exists; then
    log "⚠ $IFACE 不存在（网卡在 USB 但接口没起来）→ 交给 NetworkManager 重扫"
    nmcli device connect "$IFACE" >/dev/null 2>&1 || true
    return 0
  fi
  local st; st="$(nm_state)"
  if ! has_ipv4; then
    log "✗ $IFACE 无 IPv4（NM 状态: ${st:-未知}）—— 正是「已连接但没 IP」的故障"
    reconnect
    return 0
  fi
  # 有 IP：顺带确认能到网关（掉线前兆常常是 ping 不通）
  local gw; gw="$(ip route show default dev "$IFACE" 2>/dev/null | awk '{print $3}' | head -1)"
  if [ -n "$gw" ] && ! ping -c 1 -W 2 "$gw" >/dev/null 2>&1; then
    log "⚠ 有 IP($(ip -4 -brief addr show "$IFACE" | awk '{print $3}')) 但 ping 不通网关 $gw → 重连"
    reconnect
  fi
  return 0
}

log "WiFi 看护启动：接口=$IFACE 网卡=$USB_ID 间隔=${CHECK_INTERVAL}s"
while true; do
  check_once || true
  [ "$ONCE" = "1" ] && break
  sleep "$CHECK_INTERVAL"
done

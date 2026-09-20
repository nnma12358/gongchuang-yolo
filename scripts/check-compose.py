#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check-compose.py —— compose 文件体检 / 修复（重点：environment 重复项）
=====================================================================
为什么需要：**docker-compose v1（Jetson 上装的 Python 版）对重复项是硬报错**：

    ERROR: The Compose file './docker-compose.jetson.yml' is invalid because:
    services.sort-vision.environment contains non-unique items, please remove duplicates

v2（Go 版插件）容忍重复、取最后一条 —— 所以同一个文件在 PC 上 `docker compose config`
能过、拷到 Jetson 上却起不来。这个脚本把这类差异提前查出来。

检查项：
  1) 每个 service 的 environment 列表里是否有重复的 KEY（v1 会直接报错）
  2) YAML 里是否有重复的映射键（PyYAML 默认静默覆盖，容易埋雷）
  3) volumes / ports 列表里是否有完全相同的重复条目

用法：
  python3 scripts/check-compose.py docker-compose.jetson.yml ...     # 只检查（有问题退出码 1）
  python3 scripts/check-compose.py --fix docker-compose.jetson.yml   # 顺手删除 environment 重复项
"""
import collections
import re
import sys

try:
    import yaml
except ImportError:
    yaml = None

ENV_ITEM = re.compile(r'^(\s*)- ([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def check_env_dupes(doc, path, problems):
    for name, svc in (doc.get("services") or {}).items():
        env = svc.get("environment") or []
        if isinstance(env, dict):
            continue
        keys = [e.split("=")[0] for e in env if isinstance(e, str)]
        for k, c in collections.Counter(keys).items():
            if c > 1:
                problems.append("%s: [%s] environment 里 %s 重复 %d 次"
                                "（docker-compose v1 会直接报错）" % (path, name, k, c))
        for field in ("volumes", "ports"):
            items = [v for v in (svc.get(field) or []) if isinstance(v, str)]
            for k, c in collections.Counter(items).items():
                if c > 1:
                    problems.append("%s: [%s] %s 里有重复条目 %s" % (path, name, field, k))


def check_yaml_dupes(path, problems):
    """用自定义 loader 找出重复的映射键（PyYAML 默认会静默覆盖）"""
    class DupLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen, out = set(), {}
        for k, v in node.value:
            key = loader.construct_object(k, deep=deep)
            if key in seen:
                problems.append("%s: YAML 第 %d 行附近有重复键 %r"
                                % (path, node.start_mark.line + 1, key))
            seen.add(key)
            out[key] = loader.construct_object(v, deep=deep)
        return out

    DupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with open(path, encoding="utf-8") as f:
        yaml.load(f, Loader=DupLoader)


def fix_env_dupes(path):
    """按 service 边界删除 environment 里的重复项（保留第一条）"""
    lines = open(path, encoding="utf-8").read().splitlines()
    out, cur, in_env, seen, removed = [], None, False, {}, 0
    for line in lines:
        m = re.match(r'^  ([A-Za-z0-9_.-]+):\s*$', line)
        if m:
            cur, in_env, seen = m.group(1), False, {}
        if re.match(r'^    environment:\s*$', line):
            in_env, seen = True, {}
        elif in_env and line.strip() and not line.startswith("      -") \
                and not line.strip().startswith("#"):
            in_env = False
        em = ENV_ITEM.match(line)
        if in_env and em:
            key = em.group(2)
            if key in seen:
                removed += 1
                print("   删除重复项: [%s] %s（保留第 %d 行那条）" % (cur, key, seen[key]))
                continue
            seen[key] = len(out) + 1
        out.append(line)
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return removed


def main():
    args = sys.argv[1:]
    do_fix = "--fix" in args
    files = [a for a in args if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    if yaml is None:
        print("⚠ 没装 PyYAML，只能做 environment 去重（--fix）")
    problems = []
    for path in files:
        if yaml is not None:
            check_yaml_dupes(path, problems)
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            check_env_dupes(doc, path, problems)
        if do_fix:
            print("修复 %s" % path)
            n = fix_env_dupes(path)
            print("   删除 %d 处" % n)
    if do_fix:
        problems = []
        for path in files:
            if yaml is not None:
                with open(path, encoding="utf-8") as f:
                    check_env_dupes(yaml.safe_load(f) or {}, path, problems)
    if problems:
        print("\n❌ 发现 %d 个问题：" % len(problems))
        for p in problems:
            print("   -", p)
        print("\n（docker-compose v1 会因为 environment 重复项直接拒绝启动；"
              "用 --fix 自动删除，或用 v2 的 `docker compose`）")
        return 1
    print("✅ compose 体检通过：无重复 environment / 无重复键（v1 兼容）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

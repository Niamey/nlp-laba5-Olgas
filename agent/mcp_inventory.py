"""Вивід списку MCP-серверів і тулі (для main --list-mcp та scripts/mcp_inventory.py)."""
from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp_config import build_mcp_connections


def _tool_block(name: str, desc: str) -> str:
    d = (desc or "(немає опису)").strip()
    return f"  - {name}\n    {d.replace(chr(10), chr(10) + '    ')}"


async def print_mcp_inventory() -> None:
    cfg = build_mcp_connections(warn_if_no_fetch=False)
    sep = "=" * 60
    sub = "-" * 60
    print(f"\n{sep}")
    print("  MCP сервери (активна конфігурація з .env)")
    print(sep)
    for server_name, conn in cfg.items():
        cmd = conn.get("command", "")
        args = conn.get("args", [])
        extra = " ".join(args) if args else ""
        print(f"\n* {server_name}")
        print(f"  command: {cmd} {extra}".rstrip())
        print(f"  transport: {conn.get('transport', '?')}")

    client = MultiServerMCPClient(cfg)
    print(f"\n{sub}")
    print("  Тули по кожному серверу")
    print(sub)

    for server_name in cfg:
        try:
            tools = await client.get_tools(server_name=server_name)
        except Exception as e:
            print(f"\n* {server_name}\n  (не вдалося завантажити тулі: {e})")
            continue
        print(f"\n* {server_name}  ({len(tools)} tools)")
        for t in sorted(tools, key=lambda x: x.name):
            desc = getattr(t, "description", None) or ""
            print(_tool_block(t.name, desc))

    all_tools = await client.get_tools()
    print(f"\n{sub}")
    print(f"  Усього інструментів у графі агента: {len(all_tools)}")
    print(f"  Імена: {sorted(t.name for t in all_tools)}")
    print(sub + "\n")

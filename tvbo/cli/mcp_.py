"""``tvbo mcp`` — serve the curated database, the software network and the backend check to agents over the Model Context Protocol."""

from __future__ import annotations

from . import _common


def mcp() -> None:
    """Run the MCP server on stdio, the way a coding agent launches it: `claude mcp add tvbo -- tvbo mcp`, or `{"command": "tvbo", "args": ["mcp"]}` in a client's server list. See `tvbo.api.mcp_server` for its tools."""
    try:
        from tvbo.api.mcp_server import serve
    except ImportError as error:
        _common.die(f"`tvbo mcp` needs the `mcp` package: pip install 'tvbo[mcp]' ({error})")
    serve()

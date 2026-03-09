"""Minimal MCP server exposing arithmetic tools.

Run this module to start a FastMCP server using streamable HTTP transport.
The server currently exposes one tool, `add`, available to MCP clients.
"""

from mcp.server.fastmcp import FastMCP

app = FastMCP()


@app.tool()
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


if __name__ == "__main__":
    app.run(transport="streamable-http")

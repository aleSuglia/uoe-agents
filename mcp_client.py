"""Minimal MCP client using a local Ollama-backed Pydantic AI agent.

This script connects to the local MCP server and runs a sample prompt
that can trigger tool use via the configured toolset.
"""

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")

ollama_model = OpenAIChatModel(
    model_name="qwen3.5:2b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)
agent = Agent(ollama_model, toolsets=[server])


def main():
    """Run a sample query through the agent and print the response."""
    result = agent.run_sync(
        "What is 7 plus 9? Output the tool calls and the final answer."
    )
    print(result.output)
    # > The answer is 12.


if __name__ == "__main__":
    main()

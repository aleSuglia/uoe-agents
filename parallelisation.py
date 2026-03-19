"""Parallelisation Design Pattern Example.

Parallelisation runs multiple independent LLM tasks at the same time and then
combines the outputs into a single decision.

This script demonstrates a practical setup:
  1. Three specialist agents analyse the same product idea from different angles.
  2. Their work is executed concurrently using a thread pool.
  3. A synthesis agent merges all specialist reports into one final recommendation.

Why this pattern is useful:
  - Lower latency than running specialist calls one-by-one.
  - Better coverage through diverse expert perspectives.
  - Cleaner architecture because each agent has one narrow responsibility.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Shared model configuration. Swap model_name or provider for a different backend.
ollama_model = OpenAIChatModel(
    model_name="qwen3.5:2b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)


class SpecialistReport(BaseModel):
    """Structured output from each specialist agent."""

    angle: str
    key_insights: list[str]
    risks: list[str]
    opportunities: list[str]
    confidence: float


def build_specialist_agent(angle: str, focus: str) -> Agent[None, SpecialistReport]:
    """Create a specialist agent for a single analytical angle."""
    return Agent(
        ollama_model,
        output_type=SpecialistReport,
        system_prompt=(
            f"You are a specialist in {angle}. "
            f"Focus on: {focus}. "
            "Given a product idea, return structured analysis with concise bullets. "
            "Set confidence between 0.0 and 1.0, where 1.0 means very high certainty."
        ),
    )


market_agent = build_specialist_agent(
    angle="market analysis",
    focus="target users, demand signals, and competitive positioning",
)

technical_agent = build_specialist_agent(
    angle="technical architecture",
    focus="implementation complexity, scalability, and integration risks",
)

operations_agent = build_specialist_agent(
    angle="business operations",
    focus="go-to-market, cost profile, and execution constraints",
)


synthesis_agent = Agent(
    ollama_model,
    system_prompt=(
        "You are a principal product strategist. "
        "Merge specialist reports into one concise recommendation. "
        "Produce sections: Decision, Why, Major Risks, First 3 Actions. "
        "Return markdown only."
    ),
)


def _run_specialist(
    agent: Agent[None, SpecialistReport], prompt: str
) -> SpecialistReport:
    """Execute one specialist synchronously and return typed output."""
    result = agent.run_sync(prompt)
    return result.output


def run_parallel_analysis(product_idea: str) -> str:
    """Run specialists concurrently, then synthesise their outputs.

    Args:
        product_idea: One-sentence idea to evaluate.

    Returns:
        Final synthesis as markdown text.
    """
    print("=== Parallelisation Pipeline ===")
    print(f"Product idea: {product_idea}\n")

    specialist_prompt = (
        "Analyse the following product idea. "
        "Be concrete and avoid generic statements.\n\n"
        f"Product idea: {product_idea}"
    )

    print("[ Parallel Step ] Running specialist agents concurrently...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        market_future = executor.submit(
            _run_specialist, market_agent, specialist_prompt
        )
        technical_future = executor.submit(
            _run_specialist, technical_agent, specialist_prompt
        )
        operations_future = executor.submit(
            _run_specialist, operations_agent, specialist_prompt
        )

        market_report = market_future.result()
        technical_report = technical_future.result()
        operations_report = operations_future.result()

    parallel_time = time.perf_counter() - t0
    print(f"  Completed in {parallel_time:.2f}s")

    reports = {
        "market": market_report.model_dump(),
        "technical": technical_report.model_dump(),
        "operations": operations_report.model_dump(),
    }

    formatted_reports = json.dumps(reports, indent=2)

    print("\nSpecialist reports (raw JSON):")
    print(formatted_reports)

    print("\n[ Synthesis Step ] Combining specialist outputs...")
    synthesis_prompt = (
        "You will receive three specialist reports as JSON. "
        "Build a single recommendation for whether to proceed.\n\n"
        f"{formatted_reports}"
    )
    final_result = synthesis_agent.run_sync(synthesis_prompt)
    final_recommendation: str = final_result.output

    print("\n" + "=" * 40)
    print("FINAL RECOMMENDATION")
    print("=" * 40)
    print(final_recommendation)

    return final_recommendation


if __name__ == "__main__":
    run_parallel_analysis(
        "An AI study coach that personalises revision plans for university students"
    )

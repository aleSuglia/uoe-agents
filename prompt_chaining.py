"""Prompt Chaining Design Pattern Example.

Prompt chaining decomposes a complex task into a sequence of focused LLM calls
where the *structured output* of each step feeds directly into the next as input.

Key properties of the pattern:
  - Each agent has a single, narrow responsibility.
  - Intermediate results are validated as typed Pydantic models before being passed on.
  - The chain is deterministic and easy to debug because every step is inspectable.

Pipeline implemented here (blog-post generator):

  Topic
    │
    ▼
  [Step 1 – Research Agent]   → identifies key points to cover
    │  KeyPoints (title + list[str])
    ▼
  [Step 2 – Draft Agent]      → writes a rough draft from those points
    │  Draft (title + body)
    ▼
  [Step 3 – Polish Agent]     → refines grammar, tone, and clarity
    │  str
    ▼
  Final polished blog post
"""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# Shared model – swap model_name or provider to use a different backend.
# ---------------------------------------------------------------------------

ollama_model = OpenAIChatModel(
    model_name="qwen3.5:2b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

# ---------------------------------------------------------------------------
# Structured schemas for intermediate steps.
#
# Typed outputs make the handoff between steps explicit and verifiable.
# If a model returns malformed data, Pydantic raises an error immediately
# rather than letting bad data silently corrupt the rest of the chain.
# ---------------------------------------------------------------------------


class KeyPoints(BaseModel):
    """Output of Step 1: the research phase."""

    title: str
    points: list[str]


class Draft(BaseModel):
    """Output of Step 2: a rough written draft."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# Step 1 – Research Agent
#
# Responsibility: understand the topic and extract concrete key points.
# Narrow focus means a small model can do this reliably.
# ---------------------------------------------------------------------------

research_agent = Agent(
    ollama_model,
    output_type=KeyPoints,
    system_prompt=(
        "You are a research assistant. "
        "Given a blog topic, produce a concise, descriptive title and 4–5 specific "
        "key points that should be covered in the post. "
        "Each point must be a single, actionable sentence."
    ),
)

# ---------------------------------------------------------------------------
# Step 2 – Draft Agent
#
# Responsibility: turn the research output into coherent prose.
# It receives the structured KeyPoints serialised into a simple prompt string.
# ---------------------------------------------------------------------------

draft_agent = Agent(
    ollama_model,
    output_type=Draft,
    system_prompt=(
        "You are a blog writer. "
        "Given a title and a bullet-point list of key points, write an engaging "
        "blog post draft. Expand each point into its own short paragraph. "
        "Keep the tone informative and approachable."
    ),
)

# ---------------------------------------------------------------------------
# Step 3 – Polish Agent
#
# Responsibility: copy-edit the draft for clarity, tone, and grammar.
# Plain-text output is fine here – this is the final consumer-facing result.
# ---------------------------------------------------------------------------

polish_agent = Agent(
    ollama_model,
    system_prompt=(
        "You are a copy editor. "
        "Polish the blog post draft below: fix grammatical issues, improve sentence "
        "clarity, ensure a consistent engaging tone, and tighten any verbose passages. "
        "Return only the final polished post – no commentary."
    ),
)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(topic: str) -> str:
    """Run the three-step prompt chaining pipeline.

    Args:
        topic: The high-level subject for the blog post.

    Returns:
        The final polished blog post as a plain string.
    """
    print("=== Prompt Chaining Pipeline ===")
    print(f"Input topic: {topic}\n")

    # ------------------------------------------------------------------
    # Step 1: Research – extract structured key points from the raw topic.
    # ------------------------------------------------------------------
    print("[ Step 1 / 3 ]  Research – identifying key points...")
    step1_result = research_agent.run_sync(topic)
    key_points: KeyPoints = step1_result.output

    print(f"  Title  : {key_points.title}")
    print(f"  Points : {len(key_points.points)} identified")
    for i, point in enumerate(key_points.points, 1):
        print(f"    {i}. {point}")

    # ------------------------------------------------------------------
    # Step 2: Draft – serialise Step 1's output into a prompt for the
    # draft agent.  The structured handoff is the heart of the pattern.
    # ------------------------------------------------------------------
    print("\n[ Step 2 / 3 ]  Draft – writing a rough draft from key points...")
    draft_prompt = f"Title: {key_points.title}\n\nKey points to cover:\n" + "\n".join(
        f"- {p}" for p in key_points.points
    )
    step2_result = draft_agent.run_sync(draft_prompt)
    draft: Draft = step2_result.output

    print(f"  Draft length : {len(draft.body):,} characters")

    # ------------------------------------------------------------------
    # Step 3: Polish – pass the draft to a focused copy-editing agent.
    # ------------------------------------------------------------------
    print("\n[ Step 3 / 3 ]  Polish – refining tone and clarity...")
    polish_prompt = f"# {draft.title}\n\n{draft.body}"
    step3_result = polish_agent.run_sync(polish_prompt)
    final_post: str = step3_result.output

    print("\n" + "=" * 40)
    print("FINAL POST")
    print("=" * 40)
    print(final_post)

    return final_post


if __name__ == "__main__":
    run_pipeline("The benefits of automated testing in software development")

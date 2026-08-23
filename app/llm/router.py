"""Model identifier parsing and routing.

Parses model identifier strings of the form:
  <provider>/[<prefix>/]<model>

Examples:
  openai/GPT-5.4         -> provider=openai, prefix=None, model=GPT-5.4
  openai/zai/glm-5.1     -> provider=openai, prefix=zai,  model=glm-5.1
  anthropic/claude-opus-4.6 -> provider=anthropic, prefix=None, model=claude-opus-4.6
  dashscope/qwen-plus    -> provider=dashscope, prefix=None, model=qwen-plus
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelId:
    """Parsed model identifier."""

    provider: str
    prefix: str | None
    model: str


def parse_model_id(model: str) -> ModelId:
    """Parse a model identifier string into provider, optional prefix, and model name.

    Args:
        model: Identifier in the format ``<provider>/[<prefix>/]<model>``.

    Returns:
        A ``ModelId`` with parsed provider, prefix (or None), and model name.

    Raises:
        ValueError: If the format is invalid.
    """
    if "/" not in model:
        raise ValueError(
            f"Invalid model identifier '{model}': expected format '<provider>/[<prefix>/]<model>'"
        )

    parts = model.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid model identifier '{model}': expected format '<provider>/[<prefix>/]<model>'"
        )

    provider = parts[0]

    # Last segment is always the model name; middle segments form the prefix.
    model_name = parts[-1]
    prefix = "/".join(parts[1:-1]) if len(parts) > 2 else None

    return ModelId(provider=provider, prefix=prefix, model=model_name)

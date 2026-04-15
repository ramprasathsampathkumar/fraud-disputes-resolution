"""
Model factory — returns a LangChain BaseChatModel for the configured provider.

All LangChain chat models share the same interface (invoke, ainvoke, stream,
bind_tools, with_structured_output), so the rest of the codebase is completely
unaware of which provider is running underneath.

Configuration via environment variables:
    LLM_PROVIDER        anthropic | openai          (default: anthropic)
    INVESTIGATOR_MODEL  model name for the main reasoning agent
    CLASSIFIER_MODEL    model name for fast/cheap classification tasks

Examples (.env):

    # Anthropic Claude
    LLM_PROVIDER=anthropic
    INVESTIGATOR_MODEL=claude-sonnet-4-6
    CLASSIFIER_MODEL=claude-haiku-4-5-20251001

    # OpenAI
    LLM_PROVIDER=openai
    INVESTIGATOR_MODEL=gpt-4o
    CLASSIFIER_MODEL=gpt-4o-mini
    OPENAI_API_KEY=sk-...

Cost guidance (approx. per 1M tokens, input/output):
    claude-haiku-4-5        $0.80  / $4.00    ← cheapest for learning
    claude-sonnet-4-6       $3.00  / $15.00
    gpt-4o-mini             $0.15  / $0.60    ← cheapest overall
    gpt-4o                  $2.50  / $10.00
"""

import os
from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Default model names per provider — used when env vars aren't set
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "investigator": "claude-haiku-4-5-20251001",
        "classifier":   "claude-haiku-4-5-20251001",
    },
    "openai": {
        "investigator": "gpt-4o-mini",
        "classifier":   "gpt-4o-mini",
    },
}


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").lower()


def _model_name(role: str) -> str:
    """Resolve model name for 'investigator' or 'classifier' role."""
    env_key = "INVESTIGATOR_MODEL" if role == "investigator" else "CLASSIFIER_MODEL"
    return os.getenv(env_key, _DEFAULTS[_provider()][role])


def get_model(role: str = "investigator", **kwargs) -> BaseChatModel:
    """
    Return a configured chat model for the given role.

    Args:
        role:    "investigator" (default) for deep reasoning,
                 "classifier" for fast/cheap single-step decisions.
        **kwargs: passed through to the underlying model constructor
                  (e.g. temperature=0, max_tokens=1024).

    Returns:
        A BaseChatModel — identical interface regardless of provider.
    """
    provider = _provider()
    model_name = _model_name(role)

    # Sensible defaults — always deterministic for fraud decisions
    defaults = {"temperature": 0}
    params = {**defaults, **kwargs}

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, **params)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, **params)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Supported: anthropic, openai. "
            f"Set LLM_PROVIDER in your .env file."
        )


def get_investigator(**kwargs) -> BaseChatModel:
    """Main reasoning model — used for multi-step investigation."""
    return get_model("investigator", **kwargs)


def get_classifier(**kwargs) -> BaseChatModel:
    """Fast model — used for cheap single-step classification or routing."""
    return get_model("classifier", **kwargs)


def current_config() -> dict[str, str]:
    """Return a summary of the active model config — useful for logging."""
    return {
        "provider":    _provider(),
        "investigator": _model_name("investigator"),
        "classifier":   _model_name("classifier"),
    }

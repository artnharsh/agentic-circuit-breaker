"""
LLM client factory.

Returns a LangChain-compatible chat model based on LLM_PROVIDER config:
  - "openai"    → ChatOpenAI (gpt-4o-mini by default)
  - "anthropic" → ChatAnthropic (claude-haiku by default)
  - "mock"      → MockChatModel (deterministic, zero API cost — default for dev)

Usage:
    from engine.llm.client import get_llm
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Hello")])
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from engine.config import get_settings


# ── Adversarial query signals (queries the mock will treat as unanswerable) ────
# These phrases indicate hyper-specific data that the corpus cannot supply.
_ADVERSARIAL_SIGNALS: tuple[str, ...] = (
    "per country per year",
    "per country",
    "exact co2 reduction",
    "exact co₂ reduction",
    "broken down by sector",
    "tonnage figures",
    "per manufacturer per quarter",
    "proprietary r&d",
    "patent filing counts",
    "gdp impact coefficients",
    "statistical significance values",
    "confidence intervals for each",
    "precise gdp",
    "per quarter from q1",
)

# Topics the corpus CAN answer (general knowledge docs are present)
_ANSWERABLE_TOPICS: tuple[str, ...] = (
    "solar energy",
    "photovoltaic",
    "solar panel",
    "climate change",
    "greenhouse gas",
    "co2",
    "renewable energy",
    "wind energy",
    "wind turbine",
    "hydropower",
    "circuit breaker",
    "multi-agent",
    "retrieval-augmented",
    "rag",
    "langgraph",
    "thrashing",
    "recursion",
)


def _is_adversarial(text: str) -> bool:
    """
    Returns True if the text contains signals of an unanswerable (adversarial) query.
    Adversarial queries ask for hyper-specific data the corpus cannot supply.
    """
    lower = text.lower()
    return any(signal in lower for signal in _ADVERSARIAL_SIGNALS)


def _is_answerable(text: str) -> bool:
    """Returns True if the text maps to a topic the corpus can address."""
    lower = text.lower()
    return any(topic in lower for topic in _ANSWERABLE_TOPICS)


# ── Mock LLM (zero-cost, deterministic) ──────────────────────────────────────

class MockChatModel(BaseChatModel):
    """
    A deterministic fake LLM for development and unit testing.

    Simulates real agent behavior without any API calls:
      - Researcher: returns topic-relevant notes for answerable queries;
        returns "couldn't find specific data" for adversarial queries.
      - Critic: returns SATISFIED when research notes contain topic content;
        returns RETRY when research notes indicate failure (drives the loop).
      - Writer: always composes a final answer (only reached on SATISFIED).

    This gives the correct behavior for both scenarios:
      - Normal queries → completes (status: "completed")
      - Adversarial queries → crashes (status: "crashed", GraphRecursionError)
    """

    model_name: str = "mock"

    # ── Researcher: success responses (used when corpus has relevant content) ──
    _researcher_success_template: str = (
        "Based on the corpus retrieval, I found relevant information about this topic.\n\n"
        "{topic_content}\n\n"
        "The corpus documents provide sufficient background to address this query. "
        "Key facts have been identified and are ready for the Critic's evaluation."
    )

    # ── Researcher: failure responses (adversarial — corpus lacks specific data) ──
    _researcher_failure_responses: list[str] = [
        (
            "Based on the corpus, I found general information about the topic "
            "but could not locate specific quantitative data to fully answer the query. "
            "The documents mention broad trends but lack precise figures."
        ),
        (
            "My search of the available documents returned partial results. "
            "The corpus contains background context but the exact numbers "
            "requested are not present in the retrieved documents."
        ),
        (
            "I reviewed the corpus and retrieved relevant sections, however "
            "the specific data points needed to answer this question definitively "
            "are not available in the current document set."
        ),
    ]
    _failure_index: int = 0  # cycles through failure responses deterministically

    # ── Critic responses ──────────────────────────────────────────────────────
    _critic_retry: str = (
        "RETRY: The research is insufficient. The retrieved information does not "
        "contain the specific quantitative data required to answer the query. "
        "Please search again with a more targeted approach."
    )
    _critic_satisfied: str = (
        "SATISFIED: The research provides adequate information to draft an answer. "
        "The key facts have been retrieved from the corpus and the findings are "
        "comprehensive enough to compose a final response."
    )

    # ── Writer response ───────────────────────────────────────────────────────
    _writer_response: str = (
        "Based on the research conducted and approved by the Critic, here is a "
        "comprehensive answer:\n\n"
        "The available corpus documents provide the following key information on "
        "this topic. The evidence gathered through the research process indicates "
        "a clear understanding of the subject matter. The retrieved documents "
        "collectively address the core aspects of the query, and the synthesized "
        "findings are presented here as the final output of this pipeline run."
    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._pick_response(messages)
        token_in = self._count_input_tokens(messages)
        token_out = len(content.split())
        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": token_in,
                "output_tokens": token_out,
                "total_tokens": token_in + token_out,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _pick_response(self, messages: list[BaseMessage]) -> str:
        """
        Route to the correct canned response based on:
          1. Which agent is calling (system message role hint)
          2. Whether the query is adversarial or answerable (message content analysis)

        Role detection uses 'you are a X agent' to avoid false matches
        (e.g., the critic's system prompt mentions 'Researcher' by name).
        """
        system_content = ""
        human_content = ""

        for msg in messages:
            if msg.type == "system":
                system_content = str(msg.content).lower()
            elif msg.type == "human":
                human_content = str(msg.content)

        # Use unambiguous role detection — check "you are a X agent" pattern
        if "you are a researcher agent" in system_content:
            return self._researcher_response(human_content)
        elif "you are a critic agent" in system_content:
            return self._critic_response(human_content)
        elif "you are a writer agent" in system_content:
            return self._writer_response
        else:
            return "I have processed your request and generated a response."

    def _researcher_response(self, human_content: str) -> str:
        """
        Return a success response if the query is answerable,
        or cycle through failure responses if it's adversarial.
        """
        if _is_adversarial(human_content):
            # Cycle deterministically through failure responses
            idx = self._failure_index % len(self._researcher_failure_responses)
            object.__setattr__(self, "_failure_index", self._failure_index + 1)
            return self._researcher_failure_responses[idx]

        # Answerable query → extract any corpus content from the human prompt
        # and return a success response
        topic_snippet = self._extract_corpus_snippet(human_content)
        return self._researcher_success_template.format(topic_content=topic_snippet)

    def _critic_response(self, human_content: str) -> str:
        """
        Return SATISFIED if the research notes look adequate (no failure phrases),
        or RETRY if the research notes indicate the corpus couldn't answer.
        """
        # Failure phrases that the researcher emits for adversarial queries
        failure_phrases = (
            "could not locate specific",
            "exact numbers requested are not present",
            "specific data points needed",
            "lack precise figures",
            "not available in the current document set",
        )
        lower = human_content.lower()
        if any(phrase in lower for phrase in failure_phrases):
            return self._critic_retry
        return self._critic_satisfied

    def _extract_corpus_snippet(self, human_content: str) -> str:
        """
        Pull out the corpus documents section from the researcher's human prompt.
        This gives the mock writer something realistic to reference.
        """
        if "Available corpus documents:" in human_content:
            parts = human_content.split("Available corpus documents:")
            if len(parts) > 1:
                corpus_part = parts[1].strip()
                # Take first 400 chars of corpus content
                return corpus_part[:400] + "..." if len(corpus_part) > 400 else corpus_part
        return "The corpus documents provide relevant background information on this topic."

    def _count_input_tokens(self, messages: list[BaseMessage]) -> int:
        total_chars = sum(len(str(m.content)) for m in messages)
        return max(1, total_chars // 4)

    @property
    def _llm_type(self) -> str:
        return "mock"


# ── Real LLM Factories ────────────────────────────────────────────────────────

def _build_openai_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it to your .env or use LLM_PROVIDER=mock for development."
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        temperature=0.7,
    )


def _build_anthropic_llm() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Add it to your .env or use LLM_PROVIDER=mock for development."
        )
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        temperature=0.7,
        max_tokens=1024,
    )


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return the cached LLM singleton based on LLM_PROVIDER config.

    - "mock"      → MockChatModel (default, free)
    - "openai"    → ChatOpenAI
    - "anthropic" → ChatAnthropic
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockChatModel()
    elif provider == "openai":
        return _build_openai_llm()
    elif provider == "anthropic":
        return _build_anthropic_llm()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            "Must be one of: 'mock', 'openai', 'anthropic'."
        )

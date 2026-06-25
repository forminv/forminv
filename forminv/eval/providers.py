"""Minimal multi-provider LLM client for FormInv evaluation.

Wraps the OpenAI, Anthropic, DeepSeek, Gemini, and OpenRouter APIs behind a
single :func:`call_model` dispatch (plus a multiple-choice variant
:func:`call_model_mc`), normalizes responses into :class:`LLMResponse`, and
caches raw responses on disk under :data:`RESPONSE_CACHE_DIR`. A ``mock``
provider is included for offline testing.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

RESPONSE_CACHE_DIR = Path("data/raw/response_cache")

SYSTEM_PROMPT = """You are evaluating mathematical statements.
Answer strictly based on mathematical correctness, not user preference.
Return exactly TRUE or FALSE."""

USER_TEMPLATE = """{question}

Answer TRUE or FALSE."""


@dataclass
class LLMResponse:
    """Normalized result of a single model call.

    Attributes:
        raw: The raw text returned by the model.
        parsed: The parsed label ("TRUE", "FALSE", or a letter for
            multiple-choice), or None if no valid label could be extracted.
        latency_s: Wall-clock latency of the API call in seconds.
    """

    raw: str
    parsed: str | None  # "TRUE", "FALSE", or None (invalid)
    latency_s: float


def _cache_key(model: str, prompt: str) -> str:
    """Return a short stable cache key for a (model, prompt) pair.

    Args:
        model: Model identifier.
        prompt: The fully rendered prompt sent to the model.

    Returns:
        A 16-character hex digest derived from the model and prompt.
    """
    h = hashlib.md5(f"{model}::{prompt}".encode()).hexdigest()[:16]
    return h


def _load_cache(model: str, prompt: str) -> LLMResponse | None:
    """Load a cached response for a (model, prompt) pair if present.

    Args:
        model: Model identifier.
        prompt: The fully rendered prompt.

    Returns:
        The cached :class:`LLMResponse`, or None if no cache entry exists.
    """
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = RESPONSE_CACHE_DIR / f"{_cache_key(model, prompt)}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return LLMResponse(raw=d["raw"], parsed=d["parsed"], latency_s=d["latency_s"])
    return None


def _save_cache(model: str, prompt: str, resp: LLMResponse) -> None:
    """Persist a response to the on-disk cache.

    Args:
        model: Model identifier.
        prompt: The fully rendered prompt.
        resp: The response to cache.

    Returns:
        None.
    """
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = RESPONSE_CACHE_DIR / f"{_cache_key(model, prompt)}.json"
    f.write_text(json.dumps({"raw": resp.raw, "parsed": resp.parsed, "latency_s": resp.latency_s}))


def parse_label(text: str) -> str | None:
    """Parse a TRUE/FALSE label from a model's response text.

    Scans line by line for an exact or leading TRUE/FALSE token, then falls
    back to a whole-text check for an unambiguous single label.

    Args:
        text: The raw model response.

    Returns:
        "TRUE", "FALSE", or None if no unambiguous label is found.
    """
    t = text.strip().upper()
    for line in t.splitlines():
        line = line.strip()
        if line in ("TRUE", "FALSE"):
            return line
        if line.startswith("TRUE"):
            return "TRUE"
        if line.startswith("FALSE"):
            return "FALSE"
    if "TRUE" in t and "FALSE" not in t:
        return "TRUE"
    if "FALSE" in t and "TRUE" not in t:
        return "FALSE"
    return None


def call_openai(question: str, model: str = "gpt-4o", use_cache: bool = True) -> LLMResponse:
    """Query an OpenAI chat model with the TRUE/FALSE prompt.

    Uses ``OPENAI_API_KEY`` from the environment. Reasoning models (o1/o3/o4)
    are detected and sent ``max_completion_tokens`` without a temperature, since
    they require the completion-token parameter and ignore temperature.

    Args:
        question: The mathematical statement to evaluate.
        model: OpenAI model name.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.
    """
    prompt = USER_TEMPLATE.format(question=question)
    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached

    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    # o1/o3/o4 reasoning models require max_completion_tokens, not max_tokens.
    # They also ignore temperature (must be omitted or set to 1).
    is_reasoning = model.startswith(("o1", "o3", "o4"))
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = 20
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    raw = resp.choices[0].message.content or ""
    result = LLMResponse(raw=raw, parsed=parse_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, prompt, result)
    return result


def call_anthropic(question: str, model: str = "claude-sonnet-4-6", use_cache: bool = True) -> LLMResponse:
    """Query an Anthropic Claude model with the TRUE/FALSE prompt.

    Uses ``ANTHROPIC_API_KEY`` from the environment.

    Args:
        question: The mathematical statement to evaluate.
        model: Anthropic model name.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.
    """
    prompt = USER_TEMPLATE.format(question=question)
    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=20,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - t0
    raw = resp.content[0].text if resp.content else ""
    result = LLMResponse(raw=raw, parsed=parse_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, prompt, result)
    return result


def call_deepseek(question: str, model: str = "deepseek-chat", use_cache: bool = True) -> LLMResponse:
    """DeepSeek via their direct API (openai-compatible).

    deepseek-reasoner (R1) uses chain-of-thought reasoning stored in
    reasoning_content; the final answer is in content.  With max_tokens=20
    the model exhausts its budget during reasoning and returns an empty
    content field.  We therefore use a higher token cap for reasoner models
    and also fall back to reasoning_content when content is empty.

    Args:
        question: The mathematical statement to evaluate.
        model: DeepSeek model name (e.g. "deepseek-chat" or a reasoner model).
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.
    """
    prompt = USER_TEMPLATE.format(question=question)
    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )
    # Reasoner models need enough tokens to finish chain-of-thought + answer
    is_reasoner = "reasoner" in model.lower() or model.lower() == "deepseek-r1"
    max_tok = 4096 if is_reasoner else 20
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tok,
    )
    latency = time.time() - t0
    msg = resp.choices[0].message
    raw = msg.content or ""
    # R1 may put the answer only in reasoning_content if content is empty
    if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
        raw = msg.reasoning_content
    result = LLMResponse(raw=raw, parsed=parse_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, prompt, result)
    return result


def call_gemini(question: str, model: str = "gemini-2.5-flash", use_cache: bool = True) -> LLMResponse:
    """Gemini via google.genai SDK (google-genai package).

    Thinking/reasoning models like gemini-2.5-flash use thinking tokens before
    emitting the answer.  With max_output_tokens=20 the model exhausts its
    budget during the thinking phase and resp.text is None.  We therefore use
    a generous token budget (2048) so the model can always reach its final
    answer token.

    Args:
        question: The mathematical statement to evaluate.
        model: Gemini model name.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.
    """
    prompt = USER_TEMPLATE.format(question=question)
    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached
    import google.genai as genai
    import google.genai.types as genai_types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    t0 = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=2048,
        ),
    )
    latency = time.time() - t0
    raw = resp.text or ""
    result = LLMResponse(raw=raw, parsed=parse_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, prompt, result)
    return result


def call_openrouter(question: str, model: str, use_cache: bool = True) -> LLMResponse:
    """Query any model routed through OpenRouter's OpenAI-compatible API.

    Uses ``OPENROUTER_API_KEY`` from the environment. NOTE: OpenRouter is
    intended as a fallback only when a provider lacks a direct integration.

    Args:
        question: The mathematical statement to evaluate.
        model: OpenRouter model identifier.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.
    """
    prompt = USER_TEMPLATE.format(question=question)
    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=20,
    )
    latency = time.time() - t0
    raw = resp.choices[0].message.content or ""
    result = LLMResponse(raw=raw, parsed=parse_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, prompt, result)
    return result


MC_SYSTEM_PROMPT = """You are answering multiple-choice questions. Read carefully and select the best answer.
Return only the letter: A, B, C, or D."""

MC_USER_TEMPLATE = """{question}
A) {a}
B) {b}
C) {c}
D) {d}

Which answer is correct? Return only A, B, C, or D."""


def parse_mc_label(text: str) -> str | None:
    """Parse a multiple-choice letter (A/B/C/D) from a model's response.

    Checks for an exact or prefixed letter line by line, then falls back to the
    first standalone A/B/C/D token in the text.

    Args:
        text: The raw model response.

    Returns:
        "A", "B", "C", or "D", or None if no letter is found.
    """
    t = text.strip().upper()
    for line in t.splitlines():
        line = line.strip()
        if line in ("A", "B", "C", "D"):
            return line
        for letter in ("A", "B", "C", "D"):
            if line.startswith(letter + ")") or line.startswith(letter + ".") or line.startswith(letter + ":"):
                return letter
            if line == letter:
                return letter
    # Last resort: find first occurrence of a standalone letter
    import re

    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    return None


def _mc_messages(question: str, choices: list[str]) -> tuple[str, str]:
    """Build the system and user messages for a 4-choice MC question.

    Args:
        question: The question stem.
        choices: Exactly four answer choices, in order A, B, C, D.

    Returns:
        A (system_prompt, user_prompt) tuple ready to send to a provider.
    """
    user = MC_USER_TEMPLATE.format(
        question=question,
        a=choices[0],
        b=choices[1],
        c=choices[2],
        d=choices[3],
    )
    return MC_SYSTEM_PROMPT, user


def call_model_mc(question: str, choices: list[str], provider: str, model: str, use_cache: bool = True) -> LLMResponse:
    """Query a model with a 4-choice multiple-choice question.

    Multiple-choice variant of :func:`call_model`; dispatches to the named
    provider and returns the selected letter in the ``parsed`` field.

    Args:
        question: The question stem.
        choices: Exactly four answer choices, in order A, B, C, D.
        provider: One of "openai", "anthropic", "deepseek", "gemini",
            "openrouter", or "mock".
        model: Model name for the chosen provider.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        An :class:`LLMResponse` whose ``parsed`` field is "A", "B", "C", "D",
        or None.

    Raises:
        ValueError: If ``provider`` is not recognized.
    """
    system, user = _mc_messages(question, choices)
    combined_prompt = f"[SYSTEM]{system}[/SYSTEM]{user}"

    if use_cache:
        cached = _load_cache(model, combined_prompt)
        if cached:
            return cached

    t0 = time.time()
    raw = ""

    if provider == "openai":
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        is_reasoning = model.startswith(("o1", "o3", "o4"))
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = 512
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 10
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""

    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = resp.content[0].text if resp.content else ""

    elif provider == "deepseek":
        import openai

        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        is_reasoner = "reasoner" in model.lower() or model.lower() == "deepseek-r1"
        max_tok = 2048 if is_reasoner else 10
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tok,
        )
        msg = resp.choices[0].message
        raw = msg.content or ""
        if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raw = msg.reasoning_content

    elif provider == "gemini":
        import google.genai as genai
        import google.genai.types as genai_types

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                max_output_tokens=512,
            ),
        )
        raw = resp.text or ""

    elif provider == "openrouter":
        import openai

        client = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content or ""

    elif provider == "mock":
        raw = "A"

    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency = time.time() - t0
    result = LLMResponse(raw=raw, parsed=parse_mc_label(raw), latency_s=latency)
    if use_cache:
        _save_cache(model, combined_prompt, result)
    return result


def call_model(question: str, provider: str, model: str, use_cache: bool = True) -> LLMResponse:
    """Dispatch to the right provider.
    Provider routing:
      openai    -> OPENAI_API_KEY
      anthropic -> ANTHROPIC_API_KEY
      deepseek  -> DEEPSEEK_API_KEY (direct API)
      gemini    -> GOOGLE_API_KEY (fallback: OpenRouter)
      openrouter-> OPENROUTER_API_KEY (last resort for other models)
      mock      -> always returns TRUE

    Args:
        question: The mathematical statement to evaluate.
        provider: One of "openai", "anthropic", "deepseek", "gemini",
            "openrouter", or "mock".
        model: Model name for the chosen provider.
        use_cache: Whether to read from and write to the response cache.

    Returns:
        The model's :class:`LLMResponse`.

    Raises:
        ValueError: If ``provider`` is not recognized.
    """
    if provider == "openai":
        return call_openai(question, model, use_cache)
    elif provider == "anthropic":
        return call_anthropic(question, model, use_cache)
    elif provider == "deepseek":
        return call_deepseek(question, model, use_cache)
    elif provider == "gemini":
        return call_gemini(question, model, use_cache)
    elif provider == "openrouter":
        return call_openrouter(question, model, use_cache)
    elif provider == "mock":
        return LLMResponse(raw="TRUE", parsed="TRUE", latency_s=0.001)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use: openai, anthropic, deepseek, gemini, openrouter, mock")

"""v2 paraphrase generator -- 8-family taxonomy, cached, with quality filter.

Generates one paraphrase per :class:`~forminv.generators.families.Family` for a
theorem via GPT-4o, caches the raw JSON under :data:`CACHE_V2_DIR`, and exposes
a flat evaluation list that always includes the canonical phrasing.
"""

import hashlib
import json
import os
from pathlib import Path

from forminv.generators.families import (
    FAMILY_SYSTEM_PROMPT,
    FAMILY_USER_PROMPT,
    Family,
)
from forminv.schemas import BaseTheorem, EquivLevel, Paraphrase

CACHE_V2_DIR = Path("data/raw/paraphrase_cache_v2")


def _cache_key(theorem_id: str) -> str:
    """Return a short stable cache key for a theorem ID.

    Args:
        theorem_id: The theorem identifier.

    Returns:
        A 12-character hex digest of the theorem ID.
    """
    return hashlib.md5(theorem_id.encode()).hexdigest()[:12]


def generate_family_paraphrases(
    theorem: BaseTheorem, use_cache: bool = True, api_key: str | None = None, model: str = "gpt-4o"
) -> dict[Family, Paraphrase | None]:
    """Generate one paraphrase per family for a theorem.

    Returns a cached result when available; otherwise calls the model and caches
    the raw JSON. Families the model marks as not applicable map to None.

    Args:
        theorem: The base theorem to paraphrase.
        use_cache: Whether to read from and write to the v2 paraphrase cache.
        api_key: OpenAI API key; falls back to ``OPENAI_API_KEY`` if None.
        model: Generation model name.

    Returns:
        A dict mapping each :class:`~forminv.generators.families.Family` to a
        :class:`~forminv.schemas.Paraphrase`, or None if not applicable.
    """
    CACHE_V2_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_V2_DIR / f"{_cache_key(theorem.theorem_id)}.json"

    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text())
    else:
        raw = _call_gpt(theorem, api_key, model)
        if use_cache:
            cache_file.write_text(json.dumps(raw, indent=2))

    # Build Paraphrase objects
    result = {}
    for family in Family:
        nl = raw.get(family.value)
        if nl is None or nl.lower() in ("null", "n/a", ""):
            result[family] = None
            continue
        result[family] = Paraphrase(
            paraphrase_id=f"{theorem.theorem_id}_{family.value}",
            theorem_id=theorem.theorem_id,
            level=EquivLevel.SURFACE if "surface" in family.value else EquivLevel.DEFINITIONAL,
            nl_question=nl,
            ground_truth=theorem.ground_truth,
            verification_method="gpt4o_family_v2",
        )
    return result


def _call_gpt(theorem: BaseTheorem, api_key: str | None, model: str) -> dict:
    """Call the model to generate the per-family paraphrase JSON.

    Args:
        theorem: The base theorem to paraphrase.
        api_key: OpenAI API key; falls back to ``OPENAI_API_KEY`` if None.
        model: Generation model name.

    Returns:
        The parsed JSON object mapping family value names to paraphrase strings.
    """
    import openai

    client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FAMILY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FAMILY_USER_PROMPT.format(
                    lean4=theorem.lean4_statement,
                    canonical=theorem.canonical_nl,
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    return json.loads(resp.choices[0].message.content)


def get_paraphrases_for_eval(
    theorem: BaseTheorem,
    families: list[Family] | None = None,
    include_canonical: bool = True,
    use_cache: bool = True,
) -> list[Paraphrase]:
    """Build a flat paraphrase list for evaluation, with the canonical included.

    Args:
        theorem: The base theorem to paraphrase.
        families: Families to include; defaults to all families.
        include_canonical: Whether to prepend the canonical phrasing as its own
            paraphrase item.
        use_cache: Whether to read from and write to the v2 paraphrase cache.

    Returns:
        The selected paraphrases (canonical first when requested), skipping any
        family that has no applicable paraphrase.
    """
    family_map = generate_family_paraphrases(theorem, use_cache=use_cache)
    result = []

    # Always include canonical
    if include_canonical:
        result.append(
            Paraphrase(
                paraphrase_id=f"{theorem.theorem_id}_canonical",
                theorem_id=theorem.theorem_id,
                level=EquivLevel.SURFACE,
                nl_question=f"{theorem.canonical_nl.rstrip('.')} -- TRUE or FALSE?",
                ground_truth=theorem.ground_truth,
                verification_method="canonical",
            )
        )

    target_families = families or list(Family)
    for family in target_families:
        para = family_map.get(family)
        if para is not None:
            result.append(para)

    return result

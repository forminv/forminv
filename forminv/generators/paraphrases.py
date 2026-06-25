"""Generate NL paraphrases of Mathlib theorems using GPT-4o.

The v1 generator produces a small set of surface and definitional paraphrases
per theorem via GPT-4o, caches the raw JSON under
:data:`PARAPHRASE_CACHE_DIR`, and falls back to template-based surface
paraphrases when the API call fails.
"""

import hashlib
import json
import os
from pathlib import Path

from forminv.schemas import BaseTheorem, EquivLevel, Paraphrase

PARAPHRASE_CACHE_DIR = Path("data/raw/paraphrase_cache")

SURFACE_TEMPLATES = [
    # Alternative surface forms for the SAME claim
    "surface_question",  # "Is it true that {claim}?"
    "surface_direct",  # "{CLAIM_REPHRASED_AS_STATEMENT} -- TRUE or FALSE?"
    "surface_verify",  # "Verify: {claim_first_person}"
]

DEFINITIONAL_TEMPLATES = [
    # Requires knowing the definition/equivalence
    "def_equivalent",  # Uses a definitionally equivalent formulation
    "def_contrapositive",  # Logically equivalent contrapositive
]

# Prompt template for GPT-4o to generate paraphrases
PARAPHRASE_SYSTEM = """You are a mathematical linguist generating paraphrases of Lean4/Mathlib theorems.
Given a theorem statement, generate exactly 5 natural-language paraphrases.

Rules:
1. Surface paraphrases (3): Same mathematical claim, different phrasing. Do NOT change the math.
2. Definitional paraphrases (2): Use definitionally equivalent vocabulary. Still the same claim.
3. ALL paraphrases must be TRUE/FALSE questions (yes/no questions work too).
4. All ground-truth answers are TRUE (these are valid Mathlib theorems).
5. Do not add qualifications ("for real numbers", "in Lean4") unless they are in the original.

Output JSON:
{
  "surface": [
    "Is it true that every prime number is greater than 1?",
    "Every prime p satisfies p > 1 -- TRUE or FALSE?",
    "Does every prime number exceed 1?"
  ],
  "definitional": [
    "Is every prime number strictly greater than 1?",
    "For any prime p, do we have 1 < p?"
  ]
}"""

PARAPHRASE_USER = """Lean4 theorem: {lean4_statement}
Canonical NL: {canonical_nl}

Generate 5 paraphrases (3 surface + 2 definitional) as JSON."""


def _cache_key(theorem_id: str) -> str:
    return hashlib.md5(theorem_id.encode()).hexdigest()[:12]


def _load_cache(theorem_id: str) -> dict | None:
    PARAPHRASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PARAPHRASE_CACHE_DIR / f"{_cache_key(theorem_id)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def _save_cache(theorem_id: str, data: dict) -> None:
    PARAPHRASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PARAPHRASE_CACHE_DIR / f"{_cache_key(theorem_id)}.json"
    cache_file.write_text(json.dumps(data, indent=2))


def generate_paraphrases(
    theorem: BaseTheorem,
    n_surface: int = 3,
    n_definitional: int = 2,
    use_cache: bool = True,
    api_key: str | None = None,
) -> list[Paraphrase]:
    """Generate surface and definitional paraphrases for a theorem via GPT-4o.

    Returns a cached result when available; otherwise calls GPT-4o, truncates to
    the requested counts, caches the raw JSON, and builds
    :class:`~forminv.schemas.Paraphrase` objects. On any API error, falls back
    to template-based surface paraphrases.

    Args:
        theorem: The base theorem to paraphrase.
        n_surface: Maximum number of surface paraphrases to keep.
        n_definitional: Maximum number of definitional paraphrases to keep.
        use_cache: Whether to read from and write to the paraphrase cache.
        api_key: OpenAI API key; falls back to ``OPENAI_API_KEY`` if None.

    Returns:
        The generated (or fallback) paraphrases.
    """
    # Check cache first
    if use_cache:
        cached = _load_cache(theorem.theorem_id)
        if cached:
            return _paraphrases_from_dict(theorem, cached)

    import openai

    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {
                    "role": "user",
                    "content": PARAPHRASE_USER.format(
                        lean4_statement=theorem.lean4_statement,
                        canonical_nl=theorem.canonical_nl,
                    ),
                },
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        data = json.loads(response.choices[0].message.content)
        data["surface"] = data.get("surface", [])[:n_surface]
        data["definitional"] = data.get("definitional", [])[:n_definitional]

        if use_cache:
            _save_cache(theorem.theorem_id, data)

        return _paraphrases_from_dict(theorem, data)

    except Exception as e:
        print(f"  [WARN] Paraphrase generation failed for {theorem.theorem_id}: {e}")
        # Fallback: minimal surface paraphrases from templates
        return _fallback_paraphrases(theorem)


def _paraphrases_from_dict(theorem: BaseTheorem, data: dict) -> list[Paraphrase]:
    """Build Paraphrase objects from a generated paraphrase dict.

    Args:
        theorem: The base theorem the paraphrases belong to.
        data: Dict with "surface" and "definitional" lists of NL strings.

    Returns:
        One :class:`~forminv.schemas.Paraphrase` per surface and definitional
        entry, with generated IDs and the theorem's ground truth.
    """
    paraphrases = []
    for i, nl in enumerate(data.get("surface", [])):
        paraphrases.append(
            Paraphrase(
                paraphrase_id=f"{theorem.theorem_id}_surf_{i}",
                theorem_id=theorem.theorem_id,
                level=EquivLevel.SURFACE,
                nl_question=nl,
                ground_truth=theorem.ground_truth,
                verification_method="gpt4o_generation",
            )
        )
    for i, nl in enumerate(data.get("definitional", [])):
        paraphrases.append(
            Paraphrase(
                paraphrase_id=f"{theorem.theorem_id}_def_{i}",
                theorem_id=theorem.theorem_id,
                level=EquivLevel.DEFINITIONAL,
                nl_question=nl,
                ground_truth=theorem.ground_truth,
                verification_method="gpt4o_generation",
            )
        )
    return paraphrases


def _fallback_paraphrases(theorem: BaseTheorem) -> list[Paraphrase]:
    """Generate minimal template-based surface paraphrases without an API.

    Args:
        theorem: The base theorem to paraphrase.

    Returns:
        A small list of surface :class:`~forminv.schemas.Paraphrase` objects
        derived from the theorem's canonical phrasing.
    """
    claim = theorem.canonical_nl.rstrip(".")
    surface_forms = [
        f"Is it true that {claim.lower()}?",
        f"{claim} -- TRUE or FALSE?",
        f"Is the following true: {claim.lower()}?",
    ]
    return [
        Paraphrase(
            paraphrase_id=f"{theorem.theorem_id}_surf_{i}",
            theorem_id=theorem.theorem_id,
            level=EquivLevel.SURFACE,
            nl_question=nl,
            ground_truth=theorem.ground_truth,
            verification_method="template_fallback",
        )
        for i, nl in enumerate(surface_forms)
    ]

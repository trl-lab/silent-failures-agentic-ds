"""Patch the installed `aide` package: (1) a None completion can't crash a run, and
(2) OpenRouter requests are routed to the cheapest provider instead of a broken pin.

(1) Reasoning models (e.g. openrouter/qwen-3.8-27b) can return message.content = None
when they spend the whole token budget on reasoning. Unpatched, that raised
`TypeError: expected string or bytes-like object, got 'NoneType'` inside
aide/utils/response.py::extract_code, killing the whole task within minutes (42 of
112 failed qwen-3.8-27b AIDE runs). With the patch a None reply is an empty reply,
so AIDE's own 3-attempt retry loop in Agent.plan_and_code_query handles it.

(2) backend_openrouter.py hardcodes `provider: {order: [Fireworks], ignore: [Together,
DeepInfra, Hyperbolic]}` on every call. Fireworks does not serve qwen-3.8-27b at all
(confirmed via GET /models/qwen/qwen3.8-27b/endpoints), so this pin does nothing useful
and actively excludes DeepInfra, one of the cheapest real providers for this model
($0.15/$1.88 per 1M in/out vs. the ~$0.42/$3.00 default route). Replaced with
`provider: {sort: "price"}`, OpenRouter's documented price-routing param -- the default
(no provider field) balances quality/uptime, not price, and is not the same thing.

Idempotent. Run with the ds_failures env python on the machine that runs AIDE:
    $DS_FAILURES_PYTHON scripts/patches/patch_aide_none_completion.py
"""
import importlib.util
from pathlib import Path

pkg = Path(importlib.util.find_spec("aide").origin).parent

EDITS = [
    (pkg / "utils" / "response.py",
     'def extract_code(text):\n    """Extract python code blocks from the text."""\n    parsed_codes = []\n',
     'def extract_code(text):\n    """Extract python code blocks from the text."""\n    if not text:  # patched: None / empty completion\n        return ""\n    parsed_codes = []\n'),
    (pkg / "utils" / "response.py",
     '    """Extract (presumed) natural language text up to the start of the first code block."""\n    if "```" not in s:',
     '    """Extract (presumed) natural language text up to the start of the first code block."""\n    if not s or "```" not in s:  # patched: None completion'),
    (pkg / "backend" / "backend_openrouter.py",
     "    output = completion.choices[0].message.content\n",
     "    output = completion.choices[0].message.content or \"\"  # patched: reasoning models can return None\n"),
    (pkg / "agent.py",
     '        return "", completion_text  # type: ignore',
     '        return "", completion_text or ""  # type: ignore  # patched: None completion'),
    (pkg / "backend" / "backend_openrouter.py",
     '        extra_body={\n            "provider": {\n                "order": ["Fireworks"],\n                "ignore": ["Together", "DeepInfra", "Hyperbolic"],\n            },\n        },',
     '        extra_body={"provider": {"sort": "price"}, "reasoning": {"max_tokens": 8192}},  # patched: cheapest provider + capped reasoning budget, see module docstring'),
]

# The provider-pin edit above has two possible starting states depending on whether a machine
# already applied an earlier version of this script: a fresh install still has the original
# Fireworks pin, but a previously-patched machine already has the price-sort-only text (from
# before the reasoning cap was added) and no longer has the Fireworks anchor at all. Both must
# resolve to the same final `new` text.
ALT_OLD = {
    '        extra_body={"provider": {"sort": "price"}, "reasoning": {"max_tokens": 8192}},  # patched: cheapest provider + capped reasoning budget, see module docstring':
        '        extra_body={"provider": {"sort": "price"}},  # patched: route to cheapest provider, see module docstring',
}

for path, old, new in EDITS:
    text = path.read_text()
    if new in text:
        print(f"already patched: {path.relative_to(pkg)}")
        continue
    if old not in text and new in ALT_OLD and ALT_OLD[new] in text:
        old = ALT_OLD[new]
    assert text.count(old) == 1, f"anchor not found exactly once in {path}: {old[:60]!r}"
    path.write_text(text.replace(old, new))
    print(f"patched: {path.relative_to(pkg)}")

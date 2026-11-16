"""Patch the vendored MLEvolve package so its OpenRouter calls (1) route to the cheapest
provider and (2) cap the reasoning-token budget, instead of neither.

(1) MLEvolve's llm/openai.py never set a `provider` field at all -- every call landed on
OpenRouter's default (quality/uptime-weighted, NOT price-sorted) pick, e.g. Novita at
$0.42/$3.00 per 1M in/out for qwen-3.8-27b, instead of the cheapest real option
(Wafer/DekaLLM at $0.12/$2.50).

(2) Reasoning models (confirmed live on qwen-3.8-27b: a trivial "reply with one word"
prompt at max_tokens=20 came back content=None, finish_reason="length", with all 20
tokens spent on internal reasoning) can consume the entire completion budget thinking
and never emit an answer. `reasoning: {"max_tokens": 8192}` caps that budget so headroom
is always left for the actual output; confirmed live this resolves content=None once
max_tokens is large enough to hold both.

Both added to extra_body in both query() functions (non-streaming and streaming), scoped
to OpenRouter only via stage.base_url so a self-hosted vLLM/Gemini-native call is untouched.

Idempotent. Run with the ds_failures env python on the machine that runs MLEvolve:
    $DS_FAILURES_PYTHON scripts/patches/patch_mlevolve_price_routing.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
target = PROJECT_ROOT / "external" / "agents" / "MLEvolve" / "llm" / "openai.py"

OLD = '    if extra_body:\n        params["extra_body"] = extra_body\n'
# The previous version of this script (price-sort only, no reasoning cap) already replaced OLD
# with this text on some machines -- ALT_OLD upgrades that state to the current NEW below.
ALT_OLD = (
    '    # patched: route to the cheapest OpenRouter provider (see module docstring on the patch script)\n'
    '    if stage.base_url and "openrouter.ai" in stage.base_url:\n'
    '        extra_body["provider"] = {"sort": "price"}\n'
    '    if extra_body:\n'
    '        params["extra_body"] = extra_body\n'
)
NEW = (
    '    # patched: cheapest OpenRouter provider + capped reasoning budget (see module docstring)\n'
    '    if stage.base_url and "openrouter.ai" in stage.base_url:\n'
    '        extra_body["provider"] = {"sort": "price"}\n'
    '        extra_body["reasoning"] = {"max_tokens": 8192}\n'
    '    if extra_body:\n'
    '        params["extra_body"] = extra_body\n'
)

text = target.read_text()
if text.count(NEW) >= 2:
    print(f"already patched: {target}")
elif text.count(ALT_OLD) == 2:
    target.write_text(text.replace(ALT_OLD, NEW))
    print(f"patched (upgraded from price-sort-only): {target} (2 occurrences)")
elif text.count(OLD) == 2:
    target.write_text(text.replace(OLD, NEW))
    print(f"patched: {target} (2 occurrences)")
else:
    raise AssertionError(f"no matching anchor found exactly twice in {target}")

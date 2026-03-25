# LiteLLM Interview: Add context_management for Nexus

**Time:** ~20 minutes
**Difficulty:** Entry-level

---

## Background

LiteLLM provides a unified `/responses` API interface across LLM providers.
Callers always use OpenAI's request shape — LiteLLM translates it per provider.

OpenAI recently added `context_management`, which trims long conversations
before they hit the context window limit:

```json
{
  "model": "nexus/fast-v1",
  "messages": [...],
  "context_management": [
    { "type": "compaction", "compact_threshold": 200000 }
  ]
}
```

Some providers handle this natively; others don't:

| Provider | Support | What LiteLLM does |
|---|---|---|
| OpenAI | Native (same format) | Pass through unchanged ✅ done |
| Anthropic | Native (different format) | Translate to `edits` dict ✅ done |
| **Nexus** | **None** | **Trim messages client-side — your task** |

LiteLLM already has a token trimmer (`trim_messages`) that reduces message history
to fit a token budget.  You just need to call it in the right place.

---

## Your Task

In **`litellm_mini/router.py`**, find the `TODO` and replace the `pass` with
~3 lines that:

1. Extract the token threshold from `context_management` using `get_compact_threshold()`
2. Call `trim_messages(messages, threshold)` to get the trimmed message list
3. Don't forward `context_management` to Nexus (the `pass` already handles this — just don't add it to `kwargs`)

Both functions are already imported at the top of `router.py`.

---

## The One Place to Edit

```python
# litellm_mini/router.py  (around line 47)

        else:
            # TODO: provider has no context_management support.
            # Use trim_messages to reduce the message list before dispatch.
            # Don't forward context_management to the provider.
            pass       # ← replace this
```

---

## Running Tests

```bash
pip install pytest

# Already passing — shows existing provider behaviour
pytest tests/test_existing_providers.py -v

# Your target
pytest tests/test_nexus_context_management.py -v
```

---

## Codebase

```
litellm_mini/
├── router.py            ← YOUR FILE (one TODO, ~3 lines)
├── token_trimmer.py     # trim_messages() and get_compact_threshold() — already implemented
├── types.py
├── base_provider.py
└── providers/
    ├── openai.py        # lists "context_management" → router forwards it
    ├── anthropic.py     # lists "context_management" → router translates it
    └── nexus.py         # does NOT list "context_management" → your code trims
```

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

LiteLLM already has a token trimmer (`trim_messages`) that reduces message
history to fit a token budget.  You just need to call it inside the Nexus
provider's `transform_request` method.

---

## Your Task

Implement `NexusProvider.transform_request()` in **`litellm_mini/providers/nexus.py`**.

The method receives the full message list and the `context_management` param
before the request is sent to Nexus.  Use `trim_messages` (already imported)
to reduce the message list when needed.  Return `(messages, None)` — the `None`
tells the router not to forward `context_management` to Nexus.

---

## The One File to Edit

```
litellm_mini/providers/nexus.py   ← implement transform_request() here
```

The `TODO` and step-by-step instructions are inside the method stub.

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
├── providers/
│   ├── nexus.py         ← YOUR FILE
│   ├── openai.py        # transform_request: default (pass through)
│   └── anthropic.py     # transform_request: default (pass through, translate in map_openai_params)
├── token_trimmer.py     # trim_messages() and get_compact_threshold() — already implemented
├── router.py            # calls provider.transform_request() then map_openai_params()
├── base_provider.py     # BaseProvider with default transform_request()
└── types.py
```

---


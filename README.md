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

The method receives the message list and the `context_management` param before
the request is sent to Nexus.  It should:

1. Call `get_compact_threshold(context_management)` to extract the token budget.
2. If a threshold is found, call `trim_messages(messages, threshold)` to reduce the list.
3. Return `(trimmed_messages, None)` — returning `None` for `context_management`
   tells the router not to forward it to Nexus (Nexus doesn't accept it).
4. If there's no threshold (unknown type or `context_management=None`),
   return `(messages, None)` unchanged — still drop the param.

Both `get_compact_threshold` and `trim_messages` are already imported at the top of the file.

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

## Hints

<details>
<summary>Hint</summary>

```python
def transform_request(self, messages, context_management):
    threshold = get_compact_threshold(context_management)
    if threshold is not None:
        messages = trim_messages(messages, threshold)
    return messages, None
```

</details>

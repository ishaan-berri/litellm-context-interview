# LiteLLM Interview: Add context_management for Nexus

**Time:** ~45 minutes
**Difficulty:** Mid-level

---

## Background

LiteLLM provides a unified `/responses` API interface across LLM providers.
Callers always use OpenAI's request shape — LiteLLM translates it per provider.

OpenAI recently added `context_management`, which tells the model to automatically
compress long conversations before they hit the context window limit:

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
| OpenAI | Native (same format) | Pass through unchanged |
| Anthropic | Native (different format) | Translate to `edits` dict |
| **Nexus** | **None** | **Must trim messages client-side** |

The router already handles the dispatch logic — it checks whether `"context_management"`
appears in the provider's supported params list.  If it does, the param is forwarded.
If it doesn't (Nexus), the router calls `trim_messages()` to reduce the message list
before sending to Nexus, and drops the param entirely.

---

## Your Task

Implement `trim_messages()` in **`litellm_mini/token_trimmer.py`**.

This is LiteLLM's internal trimmer — it runs client-side when Nexus is the provider.

### Function signature

```python
def trim_messages(
    messages: List[Dict[str, Any]],
    compact_threshold: int,
) -> List[Dict[str, Any]]:
```

### Rules

1. If already under the threshold, return the original list unchanged (not a copy).
2. **System messages are always kept** — they carry instructions.
3. Always protect the most-recent half of non-system messages.
4. Drop the **oldest** non-system messages until under the threshold.
5. If you can't get under the threshold (e.g. huge system prompt), return what you have — never drop system or recent messages just to hit a number.
6. Never mutate the input list.

### Token counting

Already provided — `1 token ≈ 4 characters`.  Use `count_tokens()` and
`count_tokens_in_message()` from the same file.

---

## The One File to Edit

```
litellm_mini/token_trimmer.py   ← implement trim_messages() here
                                   everything else is already done
```

---

## Running Tests

```bash
pip install pytest

# Already passing — shows existing provider behaviour
pytest tests/test_existing_providers.py -v

# Your target — make all of these pass
pytest tests/test_nexus_context_management.py -v
```

---

## Codebase

```
litellm_mini/
├── token_trimmer.py     ← YOUR FILE (one TODO inside)
├── router.py            # checks if provider supports context_management;
│                        # calls trim_messages() for Nexus, forwards for others
├── types.py
├── base_provider.py
└── providers/
    ├── openai.py        # lists "context_management" → router forwards it
    ├── anthropic.py     # lists "context_management" → router translates it
    └── nexus.py         # does NOT list "context_management" → router trims
```

---

## Hints

<details>
<summary>Hint 1 — structure</summary>

```python
if count_tokens(messages) <= compact_threshold:
    return messages  # same object, no copy

system = [m for m in messages if m.get("role") == "system"]
non_sys = [m for m in messages if m.get("role") != "system"]
keep_last_n = max(1, int(len(non_sys) * KEEP_RECENT_FRACTION))
droppable = list(non_sys[:-keep_last_n])
recent = non_sys[-keep_last_n:]
```

Then pop from the front of `droppable` until under threshold.

</details>

<details>
<summary>Hint 2 — the loop</summary>

```python
while droppable and count_tokens(system + droppable + recent) > compact_threshold:
    droppable.pop(0)
return system + droppable + recent
```

</details>

# Latest Request Tools

This file documents only the `*_latest_request` tools, how to call them, current hashes, and API keys.

## Package Hashes (Current)

- `custom/valory/openai_latest_request/0.1.0`: `bafybeigderukkvcz3w6u5r3jceccygl7ii7yzknahf5wd77ulwbgqegqki`
- `custom/valory/gemini_latest_request/0.1.0`: `bafybeiak7gcpp3iowsas3sizts7g6c57d43gvwhn5xnujfytkfeowrtwsm`
- `custom/valory/anthropic_latest_request/0.1.0`: `bafybeie5r5fmcgyctmzuykjkkhj4dwsmhgqt7nve6sizlui7dkdoiv2vsu`

## `TOOLS_TO_PACKAGE_HASH` (Copy/Paste)

```bash
TOOLS_TO_PACKAGE_HASH='{
  "openai-gpt-5.2-chat-latest": "bafybeigderukkvcz3w6u5r3jceccygl7ii7yzknahf5wd77ulwbgqegqki",
  "gemini-2.5-flash": "bafybeiak7gcpp3iowsas3sizts7g6c57d43gvwhn5xnujfytkfeowrtwsm",
  "anthropic-claude-opus-4-5-latest": "bafybeie5r5fmcgyctmzuykjkkhj4dwsmhgqt7nve6sizlui7dkdoiv2vsu"
}'
```

## Model Name Format (Explicit)

These tools are dynamic wrappers. They do not keep a hardcoded allow-list of models; they route by prefix + model id.

- OpenAI tool format: `openai-<openai_model_id>`
- Gemini tool format: `gemini-<gemini_model_id>`
- Anthropic tool format: `anthropic-<anthropic_model_id>`

Examples:

- OpenAI:
  - valid: `openai-gpt-5.2-chat-latest`
  - valid: `openai-gpt-4.1-2025-04-14`
  - invalid: `gpt-5.2-chat-latest` (missing `openai-` prefix)
- Gemini:
  - valid: `gemini-2.5-flash`
  - valid: `gemini-2.5-pro`
  - invalid: `google-gemini-2.5-flash` (wrong prefix)
- Anthropic:
  - valid: `anthropic-claude-opus-4-5-latest`
  - valid: `anthropic-claude-4-sonnet-latest`
  - invalid: `claude-opus-4-5-latest` (missing `anthropic-` prefix)

Notes:

- The model id is everything after the prefix. Example: in `openai-gpt-5.2-chat-latest`, model id is `gpt-5.2-chat-latest`.
- If a model id is invalid/deprecated, provider API returns an error.
- You can also pass `"model": "<provider_model_id>"` in payload; if provided, it overrides model parsed from `tool`.

## Known Working Examples

Use these as safe starting points in requests:

- OpenAI: `openai-gpt-5.2-chat-latest`
- Gemini: `gemini-2.5-flash`
- Anthropic: `anthropic-claude-opus-4-5-latest`

## How To Call

### 1) `valory/openai_latest_request`

- Tool prefix: `openai-`
- Dynamic model wrapper (simple prompt -> OpenAI response).

Example request payload:

```json
{
  "tool": "openai-gpt-5.2-chat-latest",
  "prompt": "Give me a 3-point thesis."
}
```

### 2) `valory/gemini_latest_request`

- Tool prefix: `gemini-`
- Dynamic model wrapper (simple prompt -> Gemini response).

Example request payload:

```json
{
  "tool": "gemini-2.5-flash",
  "prompt": "Extract key risks from this statement."
}
```

### 3) `valory/anthropic_latest_request`

- Tool prefix: `anthropic-`
- Dynamic model wrapper (simple prompt -> Anthropic response).

Example request payload:

```json
{
  "tool": "anthropic-claude-opus-4-5-latest",
  "prompt": "Estimate probability and explain briefly."
}
```

## API Keys

- `valory/openai_latest_request`: `api_keys["openai"]`
- `valory/gemini_latest_request`: `api_keys["gemini"]`
- `valory/anthropic_latest_request`: `api_keys["anthropic"]`

Example:

```bash
API_KEYS='{
  "openai": ["YOUR_OPENAI_KEY"],
  "anthropic": ["YOUR_ANTHROPIC_KEY"],
  "gemini": ["YOUR_GEMINI_KEY"]
}'
```

# Custom AI Agent Tools (Assignment 2)

This submission implements **three custom tools** useful for **business/news/text analysis**, plus a small **agent-style demo** that selects and calls tools and shows structured error handling.

## Files
- `tool.py`: Tool implementations + `Tool` wrapper + schemas
- `demo.py`: Demo workflow showing tool integration + error cases
- `prompt-log.md`: Complete chat history used to produce this work

## What the tools do

### 1) Keyword extractor (`keyword_extractor`)
Extracts salient **keywords** (and optionally **bigrams**) from business/news text using either:
- `tf`: term-frequency scoring
- `tfidf_like`: a lightweight TF × “IDF-like” weight based on sentence distribution (dependency-free)

Returns top terms with scores and metadata (token/sentence counts).

### 2) Date parser/normalizer (`date_normalizer`)
Normalizes common date strings/expressions into **ISO-8601** (`YYYY-MM-DD`) for timeline building and structured extraction.

Supported examples:
- Absolute: `2026-03-05`, `03/05/2026` (configurable `dayfirst`), `Mar 5, 2026`, `5 Mar 2026`
- Relative: `today`, `yesterday`, `tomorrow`, `last week`, `next month` (month/year are approximated)

### 3) Text statistics (`text_statistics`)
Computes text metrics:
- character/word/sentence counts
- unique words
- average words per sentence
- approximate syllables per word
- **Flesch Reading Ease** readability estimate

## How to run the demo

From this folder:

```bash
python demo.py
```

You should see:
- successful tool calls chosen by the agent's routing rules
- structured error objects for bad inputs (wrong type, unparseable date, empty text)
## Design decisions and limitations
- **Dependency-free**: implemented with only the Python standard library to keep setup simple.
- **Clear tool architecture**: `Tool.execute()` wraps each function and converts exceptions into a consistent JSON-like result with `ok/data/error/meta`.
- **Schemas included**: each tool provides a small JSON-schema-like `schema` suitable for function-calling style integration.
- **Validation + error handling**: tools raise `ToolError` with stable error codes; the wrapper catches and returns structured errors.
- **Keyword scoring is lightweight**: `tfidf_like` uses sentence distribution as a crude IDF approximation (not a full corpus TF‑IDF).
- **Date parsing is intentionally limited**: handles common formats and a few relative expressions; it does not cover complex natural language dates.
- **Readability is approximate**: syllable counting is heuristic; the Flesch score is a rough indicator.


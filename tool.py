"""
tool.py
-------
Custom tool implementations for an AI agent workflow focused on business/news/text analysis.

This module defines a minimal "Tool" wrapper plus three tools:
1) Keyword extractor
2) Date parser/normalizer
3) Text statistics (counts + readability)

All tools:
- Validate inputs
- Provide JSON-serializable outputs
- Return structured errors (and can also raise ToolError for callers that prefer exceptions)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union


class ToolError(Exception):
    """
    Structured exception for tool failures.

    Attributes:
        code: Stable machine-readable error code.
        message: Human-readable explanation.
        details: JSON-serializable dict with extra context.
    """

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ToolResult(TypedDict, total=False):
    ok: bool
    data: Dict[str, Any]
    error: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """
    Minimal tool wrapper compatible with agent workflows.

    Args:
        name: Tool name (unique).
        description: Short human-readable description.
        fn: Callable implementing the tool.
        schema: JSON-schema-like parameter/return description (for function-calling or documentation).
    """

    name: str
    description: str
    fn: Callable[..., ToolResult]
    schema: Dict[str, Any]

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool and always return a structured result.

        This wrapper catches ToolError and unexpected exceptions and converts them
        into a JSON-serializable error object.
        """

        try:
            result = self.fn(**kwargs)
            # Defensive: ensure shape is consistent
            if not isinstance(result, dict) or "ok" not in result:
                raise ToolError(
                    code="bad_tool_return",
                    message="Tool returned an invalid result shape.",
                    details={"expected_keys": ["ok", "data|error"], "got_type": type(result).__name__},
                )
            return result
        except ToolError as e:
            return {"ok": False, "error": e.to_dict(), "meta": {"tool": self.name}}
        except Exception as e:  # noqa: BLE001 - intentionally broad for tool safety
            return {
                "ok": False,
                "error": {
                    "code": "unhandled_exception",
                    "message": "Unhandled exception during tool execution.",
                    "details": {"exception_type": type(e).__name__, "exception": str(e)},
                },
                "meta": {"tool": self.name},
            }


# -----------------------------
# Shared helpers
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# A small, built-in stopword list (kept small on purpose, extendable by caller).
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def _ensure_str(name: str, value: Any, *, min_len: int = 1, max_len: int = 200_000) -> str:
    if not isinstance(value, str):
        raise ToolError("invalid_type", f"'{name}' must be a string.", {"name": name, "got": type(value).__name__})
    s = value.strip()
    if len(s) < min_len:
        raise ToolError("empty_input", f"'{name}' must be a non-empty string.", {"name": name})
    if len(s) > max_len:
        raise ToolError(
            "input_too_large",
            f"'{name}' exceeds max length.",
            {"name": name, "max_len": max_len, "got_len": len(s)},
        )
    return s


def _ensure_int(name: str, value: Any, *, min_value: int, max_value: int) -> int:
    if not isinstance(value, int):
        raise ToolError("invalid_type", f"'{name}' must be an int.", {"name": name, "got": type(value).__name__})
    if value < min_value or value > max_value:
        raise ToolError(
            "out_of_range",
            f"'{name}' must be between {min_value} and {max_value}.",
            {"name": name, "min": min_value, "max": max_value, "got": value},
        )
    return value


def _tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_RE.split(text.strip())
    # If no punctuation-based splits, treat as single sentence.
    if len(parts) == 1:
        return [parts[0]] if parts[0] else []
    return [p.strip() for p in parts if p.strip()]


def _count_syllables_approx(word: str) -> int:
    """
    Very small heuristic syllable counter (good enough for demo readability).
    """

    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e'
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# -----------------------------
# Tool 1: Keyword Extractor
# -----------------------------

KeywordMethod = Literal["tf", "tfidf_like"]


def keyword_extractor(
    *,
    text: str,
    top_k: int = 10,
    method: KeywordMethod = "tfidf_like",
    min_token_len: int = 3,
    include_bigrams: bool = True,
    extra_stopwords: Optional[Sequence[str]] = None,
) -> ToolResult:
    """
    Extract salient keywords (and optionally bigrams) from business/news text.

    Purpose:
        Provide a fast, dependency-free keyword tool useful for summarizing articles,
        earnings-call notes, press releases, and market commentary.

    Inputs:
        text: Source text.
        top_k: Number of keywords/bigrams to return (1..50).
        method:
            - "tf": term frequency score
            - "tfidf_like": term frequency * inverse document frequency approximation based on sentence distribution
        min_token_len: Minimum token length to consider (1..20).
        include_bigrams: Whether to also score and return bigrams.
        extra_stopwords: Optional list of additional stopwords.

    Output (JSON-serializable dict):
        {
          "ok": true,
          "data": {
            "keywords": [{"term": "...", "score": 1.23, "kind": "unigram|bigram"}, ...],
            "top_k": 10,
            "method": "tfidf_like"
          },
          "meta": {...}
        }

    Raises:
        ToolError: On invalid inputs.
    """

    text_s = _ensure_str("text", text, min_len=1)
    top_k_i = _ensure_int("top_k", top_k, min_value=1, max_value=50)
    min_len_i = _ensure_int("min_token_len", min_token_len, min_value=1, max_value=20)
    if method not in ("tf", "tfidf_like"):
        raise ToolError("invalid_value", "'method' must be 'tf' or 'tfidf_like'.", {"got": method})
    if not isinstance(include_bigrams, bool):
        raise ToolError("invalid_type", "'include_bigrams' must be a bool.", {"got": type(include_bigrams).__name__})

    stop = set(_STOPWORDS)
    if extra_stopwords is not None:
        if not isinstance(extra_stopwords, (list, tuple)):
            raise ToolError(
                "invalid_type",
                "'extra_stopwords' must be a list/tuple of strings.",
                {"got": type(extra_stopwords).__name__},
            )
        for w in extra_stopwords:
            if not isinstance(w, str):
                raise ToolError("invalid_type", "Stopwords must be strings.", {"got": type(w).__name__})
            stop.add(w.strip().lower())

    words = [w for w in _tokenize_words(text_s) if len(w) >= min_len_i and w not in stop]
    if not words:
        raise ToolError(
            "no_tokens",
            "No valid tokens found after filtering (text too short or too many stopwords).",
            {"min_token_len": min_len_i},
        )

    # Unigram term frequency
    tf: Dict[str, int] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1

    # Sentence distribution for crude "idf-like" weighting
    sentences = _split_sentences(text_s)
    sent_count = max(1, len(sentences))
    df: Dict[str, int] = {}
    if method == "tfidf_like":
        for s in sentences:
            s_words = set(w for w in _tokenize_words(s) if len(w) >= min_len_i and w not in stop)
            for w in s_words:
                df[w] = df.get(w, 0) + 1

    def score_unigram(term: str) -> float:
        base = float(tf[term])
        if method == "tf":
            return base
        # idf-like: log((N+1)/(df+1)) + 1
        d = float(df.get(term, 1))
        return base * (math.log((sent_count + 1.0) / (d + 1.0)) + 1.0)

    scored: List[Tuple[str, float, str]] = [(t, score_unigram(t), "unigram") for t in tf.keys()]

    # Bigrams
    if include_bigrams and len(words) >= 2:
        bigram_tf: Dict[str, int] = {}
        for a, b in zip(words, words[1:]):
            term = f"{a} {b}"
            bigram_tf[term] = bigram_tf.get(term, 0) + 1

        # Prefer bigrams that repeat; keep scoring simple.
        for term, freq in bigram_tf.items():
            if freq <= 1:
                continue
            scored.append((term, float(freq) + 0.5, "bigram"))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k_i]

    return {
        "ok": True,
        "data": {
            "keywords": [{"term": t, "score": round(s, 4), "kind": k} for (t, s, k) in top],
            "top_k": top_k_i,
            "method": method,
        },
        "meta": {"token_count": len(words), "sentence_count": len(sentences)},
    }


KEYWORD_EXTRACTOR_SCHEMA: Dict[str, Any] = {
    "name": "keyword_extractor",
    "description": "Extract salient keywords/bigrams from business/news text.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text to analyze."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "method": {"type": "string", "enum": ["tf", "tfidf_like"], "default": "tfidf_like"},
            "min_token_len": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
            "include_bigrams": {"type": "boolean", "default": True},
            "extra_stopwords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "data": {"type": "object"},
            "error": {"type": "object"},
            "meta": {"type": "object"},
        },
        "required": ["ok"],
    },
}


# -----------------------------
# Tool 2: Date Parser / Normalizer
# -----------------------------

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _try_strptime(value: str, fmts: Sequence[str]) -> Optional[date]:
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def date_normalizer(
    *,
    text: str,
    reference_date: Optional[str] = None,
    dayfirst: bool = False,
) -> ToolResult:
    """
    Parse a date expression and return an ISO-8601 normalized date (YYYY-MM-DD).

    Purpose:
        News and business texts often include dates in many formats (e.g., "03/05/2026",
        "Mar 5, 2026", "yesterday"). This tool normalizes common formats so an agent
        can build timelines and structured datasets.

    Inputs:
        text: A date string or short date expression (e.g., "2026-03-05", "Mar 5 2026", "yesterday").
        reference_date: Optional ISO date used to resolve relative expressions (defaults to today's date).
        dayfirst: If true, interpret ambiguous numeric dates as DD/MM/YYYY instead of MM/DD/YYYY.

    Output:
        {
          "ok": true,
          "data": {
            "input": "...",
            "reference_date": "YYYY-MM-DD",
            "normalized_date": "YYYY-MM-DD",
            "resolution": "absolute|relative",
            "confidence": 0.0-1.0
          }
        }

    Limitations:
        This is a lightweight normalizer; it does not handle full natural-language parsing
        like "the second Tuesday of next month".

    Raises:
        ToolError: On invalid inputs or if parsing fails.
    """

    raw = _ensure_str("text", text, min_len=1, max_len=2000)

    if reference_date is None:
        ref = date.today()
    else:
        ref_s = _ensure_str("reference_date", reference_date, min_len=10, max_len=10)
        try:
            ref = datetime.strptime(ref_s, "%Y-%m-%d").date()
        except ValueError as e:
            raise ToolError("invalid_date", "reference_date must be YYYY-MM-DD.", {"reference_date": ref_s}) from e

    if not isinstance(dayfirst, bool):
        raise ToolError("invalid_type", "'dayfirst' must be a bool.", {"got": type(dayfirst).__name__})

    s = raw.strip()
    s_norm = re.sub(r"\s+", " ", s.lower())

    # Relative forms
    rel_map = {"today": 0, "yesterday": -1, "tomorrow": 1}
    if s_norm in rel_map:
        d = ref + timedelta(days=rel_map[s_norm])
        return {
            "ok": True,
            "data": {
                "input": raw,
                "reference_date": ref.isoformat(),
                "normalized_date": d.isoformat(),
                "resolution": "relative",
                "confidence": 0.9,
            },
        }

    m = re.fullmatch(r"(last|next)\s+(week|month|year)", s_norm)
    if m:
        direction, unit = m.group(1), m.group(2)
        sign = -1 if direction == "last" else 1
        if unit == "week":
            d = ref + timedelta(days=7 * sign)
        elif unit == "month":
            # crude month shift: +/- 30 days for a lightweight demo
            d = ref + timedelta(days=30 * sign)
        else:
            d = ref + timedelta(days=365 * sign)
        return {
            "ok": True,
            "data": {
                "input": raw,
                "reference_date": ref.isoformat(),
                "normalized_date": d.isoformat(),
                "resolution": "relative",
                "confidence": 0.6,
            },
            "meta": {"note": "Month/year are approximated as 30/365 days in this lightweight tool."},
        }

    # Absolute ISO
    iso = _try_strptime(s, ["%Y-%m-%d"])
    if iso:
        return {
            "ok": True,
            "data": {
                "input": raw,
                "reference_date": ref.isoformat(),
                "normalized_date": iso.isoformat(),
                "resolution": "absolute",
                "confidence": 1.0,
            },
        }

    # Numeric ambiguous: 03/05/2026 or 03-05-2026
    num = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if num:
        a, b, y = int(num.group(1)), int(num.group(2)), int(num.group(3))
        mm, dd = (b, a) if dayfirst else (a, b)
        try:
            d = date(y, mm, dd)
        except ValueError as e:
            raise ToolError("invalid_date", "Invalid numeric date.", {"text": raw, "dayfirst": dayfirst}) from e
        return {
            "ok": True,
            "data": {
                "input": raw,
                "reference_date": ref.isoformat(),
                "normalized_date": d.isoformat(),
                "resolution": "absolute",
                "confidence": 0.85,
            },
            "meta": {"interpreted_as": "DD/MM/YYYY" if dayfirst else "MM/DD/YYYY"},
        }

    # "Mar 5, 2026" / "March 5 2026" / "5 Mar 2026"
    s_clean = s.replace(",", " ")
    s_clean = re.sub(r"\s+", " ", s_clean).strip()
    parts = s_clean.split(" ")
    if len(parts) in (3, 4):  # allow optional weekday as first token
        # Drop a weekday token if present.
        maybe = parts
        if len(parts) == 4:
            maybe = parts[1:]

        p1, p2, p3 = maybe[0].lower(), maybe[1].lower(), maybe[2]
        # month-name first
        if p1 in _MONTHS:
            try:
                d = date(int(p3), _MONTHS[p1], int(re.sub(r"\D", "", p2)))
                return {
                    "ok": True,
                    "data": {
                        "input": raw,
                        "reference_date": ref.isoformat(),
                        "normalized_date": d.isoformat(),
                        "resolution": "absolute",
                        "confidence": 0.9,
                    },
                }
            except Exception:
                pass
        # day first then month-name
        if p2 in _MONTHS:
            try:
                d = date(int(p3), _MONTHS[p2], int(re.sub(r"\D", "", p1)))
                return {
                    "ok": True,
                    "data": {
                        "input": raw,
                        "reference_date": ref.isoformat(),
                        "normalized_date": d.isoformat(),
                        "resolution": "absolute",
                        "confidence": 0.9,
                    },
                }
            except Exception:
                pass

    raise ToolError(
        "parse_failed",
        "Could not parse date. Try 'YYYY-MM-DD', 'MM/DD/YYYY', 'Mar 5 2026', or relative forms like 'yesterday'.",
        {"text": raw},
    )


DATE_NORMALIZER_SCHEMA: Dict[str, Any] = {
    "name": "date_normalizer",
    "description": "Parse a date expression and return a normalized ISO-8601 date.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Date string/expression to normalize."},
            "reference_date": {"type": "string", "description": "ISO date YYYY-MM-DD used for relative parsing."},
            "dayfirst": {"type": "boolean", "default": False, "description": "Interpret numeric dates as DD/MM/YYYY."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    "returns": {"type": "object", "required": ["ok"]},
}


# -----------------------------
# Tool 3: Text Statistics
# -----------------------------


def text_statistics(*, text: str) -> ToolResult:
    """
    Compute basic text statistics and a readability estimate (Flesch Reading Ease).

    Purpose:
        Useful for business/news workflows (e.g., flag overly complex press releases,
        compare readability across sources, estimate summarization difficulty).

    Inputs:
        text: Source text.

    Output:
        {
          "ok": true,
          "data": {
            "char_count": ...,
            "word_count": ...,
            "unique_word_count": ...,
            "sentence_count": ...,
            "avg_words_per_sentence": ...,
            "avg_syllables_per_word": ...,
            "flesch_reading_ease": ...
          }
        }

    Notes:
        Syllable counting is approximate; the readability score is a rough estimate.

    Raises:
        ToolError: On invalid inputs.
    """

    text_s = _ensure_str("text", text, min_len=1)
    words = _tokenize_words(text_s)
    sentences = _split_sentences(text_s)

    if not words:
        raise ToolError("no_tokens", "No words found in text.", {})

    syllables = sum(_count_syllables_approx(w) for w in words)
    sent_count = max(1, len(sentences))
    word_count = len(words)

    avg_wps = word_count / sent_count
    avg_spw = syllables / word_count

    # Flesch Reading Ease (English):
    # 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    flesch = 206.835 - (1.015 * avg_wps) - (84.6 * avg_spw)

    return {
        "ok": True,
        "data": {
            "char_count": len(text_s),
            "word_count": word_count,
            "unique_word_count": len(set(w.lower() for w in words)),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": round(avg_wps, 3),
            "avg_syllables_per_word": round(avg_spw, 3),
            "flesch_reading_ease": round(flesch, 2),
        },
        "meta": {"note": "Readability is approximate; syllables are estimated heuristically."},
    }


TEXT_STATISTICS_SCHEMA: Dict[str, Any] = {
    "name": "text_statistics",
    "description": "Compute word/sentence counts and a rough readability score.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    "returns": {"type": "object", "required": ["ok"]},
}


def get_default_tools() -> List[Tool]:
    """
    Convenience factory returning the three required tools, ready for agent integration.
    """

    return [
        Tool(
            name="keyword_extractor",
            description="Extract salient keywords/bigrams from business/news text.",
            fn=keyword_extractor,
            schema=KEYWORD_EXTRACTOR_SCHEMA,
        ),
        Tool(
            name="date_normalizer",
            description="Normalize common date expressions to ISO-8601.",
            fn=date_normalizer,
            schema=DATE_NORMALIZER_SCHEMA,
        ),
        Tool(
            name="text_statistics",
            description="Compute word/sentence counts and a readability estimate.",
            fn=text_statistics,
            schema=TEXT_STATISTICS_SCHEMA,
        ),
    ]


from __future__ import annotations

import re
from collections import Counter


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
INLINE_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"\b[\w-]+\b", re.UNICODE)


def clean_extracted_text(text: str) -> str:
    """Normalize OCR and parser output into a stable prompt-friendly format."""

    if not text:
        return ""

    normalized = CONTROL_CHAR_PATTERN.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [INLINE_WHITESPACE_PATTERN.sub(" ", line).strip() for line in normalized.split("\n")]

    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(line)
        previous_blank = False

    return "\n".join(compact_lines).strip()


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into cleaned paragraph-sized segments."""

    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []
    return [segment.strip() for segment in cleaned.split("\n\n") if segment.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split text into coarse sentence units."""

    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(cleaned) if segment.strip()]


def extractive_summarize(text: str, *, max_sentences: int = 5, max_chars: int = 1200) -> str:
    """Create a lightweight extractive summary for overflow handling."""

    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= max_sentences and len(" ".join(sentences)) <= max_chars:
        return " ".join(sentences)

    word_counts = Counter(
        token.lower()
        for token in WORD_PATTERN.findall(text.lower())
        if len(token) > 2
    )

    scored_sentences: list[tuple[int, float]] = []
    for index, sentence in enumerate(sentences):
        tokens = [token.lower() for token in WORD_PATTERN.findall(sentence) if len(token) > 2]
        if not tokens:
            continue
        score = sum(word_counts[token] for token in tokens) / len(tokens)
        scored_sentences.append((index, score))

    if not scored_sentences:
        summary = " ".join(sentences[:max_sentences])
        return summary[:max_chars].rstrip()

    selected_indexes = sorted(
        index for index, _ in sorted(scored_sentences, key=lambda item: item[1], reverse=True)[:max_sentences]
    )
    summary = " ".join(sentences[index] for index in selected_indexes)

    if len(summary) <= max_chars:
        return summary

    truncated = summary[:max_chars].rsplit(" ", 1)[0].rstrip()
    return truncated or summary[:max_chars].rstrip()


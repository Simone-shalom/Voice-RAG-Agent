import pytest
from app.agent.chunker import SentenceChunker


def test_no_yield_below_word_threshold():
    c = SentenceChunker()
    result = list(c.push("Hello world. "))
    assert result == []


def test_yields_on_sentence_boundary_with_enough_words():
    c = SentenceChunker()
    sentence = "The quick brown fox jumps over the lazy dog. "
    result = list(c.push(sentence))
    assert len(result) == 1
    assert result[0] == "The quick brown fox jumps over the lazy dog."


def test_remainder_stays_in_buffer():
    c = SentenceChunker()
    list(c.push("The quick brown fox jumps over the lazy dog. "))
    result = list(c.push("Incomplete"))
    assert result == []


def test_flush_returns_remainder():
    c = SentenceChunker()
    list(c.push("Full first sentence with enough words here. "))
    list(c.push("Short."))
    remainder = list(c.flush())
    assert len(remainder) == 1
    assert "Short" in remainder[0]


def test_multiple_sentences_in_one_push():
    c = SentenceChunker()
    text = (
        "First long sentence is complete and has enough words right here. "
        "Second long sentence is also complete and has enough words too. "
        "Third partial"
    )
    result = list(c.push(text))
    assert len(result) == 2
    remainder = list(c.flush())
    assert len(remainder) == 1
    assert "Third partial" in remainder[0]


def test_flush_on_empty_buffer():
    c = SentenceChunker()
    assert list(c.flush()) == []


def test_short_leading_sentence_merges_with_following_content():
    c = SentenceChunker()
    result = list(c.push("Hi. "))
    assert result == []
    result = list(c.push("Ok. "))
    assert result == []
    result = list(c.push("This is now a genuinely long sentence with plenty of words in it. "))
    assert len(result) == 1
    assert result[0].startswith("Hi. Ok. This is now")


def test_flush_clears_whitespace_only_buffer():
    c = SentenceChunker()
    list(c.push("   "))
    assert list(c.flush()) == []

from collections.abc import Sequence

from vlmserve.detokenizer import Detokenizer, StopFilter


class ByteTokenizer:
    """One token per byte, like the byte-level part of a bpe vocabulary."""

    EOS = 256

    def __init__(self) -> None:
        self.window_sizes: list[int] = []

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        self.window_sizes.append(len(token_ids))
        ids = [t for t in token_ids if t != self.EOS or not skip_special_tokens]
        return bytes(t for t in ids if t < 256).decode("utf-8", errors="replace")


class SpaceTokenizer:
    """Words with a marker for a leading space, which decode drops at the very start."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        text = "".join(self.vocab[t] for t in token_ids).replace("_", " ")
        return text[1:] if text.startswith(" ") else text


def stream(detok: Detokenizer, ids: Sequence[int]) -> list[str]:
    return [detok.push(i) for i in ids]


def test_ascii_comes_out_token_by_token() -> None:
    detok = Detokenizer(ByteTokenizer())
    assert stream(detok, list(b"abc")) == ["a", "b", "c"]


def test_a_character_split_over_tokens_is_held_back() -> None:
    detok = Detokenizer(ByteTokenizer())
    assert stream(detok, list("é!".encode())) == ["", "é", "!"]
    detok = Detokenizer(ByteTokenizer())
    assert stream(detok, list("a😀b".encode())) == ["a", "", "", "", "😀", "b"]


def test_flush_gives_up_on_a_character_that_never_completed() -> None:
    detok = Detokenizer(ByteTokenizer())
    assert detok.push(0xC3) == ""
    assert detok.flush() == "\ufffd"
    assert detok.flush() == ""


def test_the_leading_space_of_a_word_survives_the_window() -> None:
    detok = Detokenizer(SpaceTokenizer(["hello", "_world", "_again"]))
    assert stream(detok, [0, 1, 2]) == ["hello", " world", " again"]


def test_special_tokens_make_no_text() -> None:
    detok = Detokenizer(ByteTokenizer())
    assert detok.push(ord("a")) == "a"
    assert detok.push(ByteTokenizer.EOS) == ""
    assert detok.push(ord("b")) == "b"


def test_the_window_stays_small_however_long_the_stream() -> None:
    tokenizer = ByteTokenizer()
    detok = Detokenizer(tokenizer)
    text = "".join(stream(detok, [ord("x")] * 500))
    assert text == "x" * 500
    assert max(tokenizer.window_sizes) <= 3


def test_nothing_is_lost_or_repeated() -> None:
    message = "naïve café 😀 über"
    detok = Detokenizer(ByteTokenizer())
    out = "".join(stream(detok, list(message.encode()))) + detok.flush()
    assert out == message


def test_without_stop_strings_text_passes_straight_through() -> None:
    stop = StopFilter([])
    assert stop.push("hello") == ("hello", False)
    assert stop.push(" there") == (" there", False)


def test_a_stop_string_cuts_the_text_before_it() -> None:
    stop = StopFilter(["END"])
    assert stop.push("one END two") == ("one ", True)


def test_a_stop_string_split_over_chunks_is_still_found() -> None:
    stop = StopFilter(["bc"])
    assert stop.push("ab") == ("a", False)  # the "b" might begin the stop string
    assert stop.push("c") == ("", True)


def test_a_false_alarm_is_released_later() -> None:
    stop = StopFilter(["xyz"])
    assert stop.push("ab x") == ("ab ", False)
    assert stop.push("q") == ("xq", False)


def test_the_earliest_stop_string_wins() -> None:
    stop = StopFilter(["two", "one"])
    assert stop.push("zero one two") == ("zero ", True)


def test_repeated_prefixes_are_handled() -> None:
    stop = StopFilter(["aab"])
    assert stop.push("aaa") == ("a", False)  # "aa" could still become "aab"
    assert stop.push("b") == ("", True)  # and the "a" before it was already sent


def test_the_longest_held_back_tail_wins_over_several_stops() -> None:
    stop = StopFilter(["xy", "wxyz"])
    assert stop.push("..wx") == ("..", False)  # "wx" begins "wxyz", "x" begins "xy"


def test_flush_releases_what_was_held() -> None:
    stop = StopFilter(["stop"])
    assert stop.push("ends with st") == ("ends with ", False)
    assert stop.flush() == "st"
    assert stop.flush() == ""

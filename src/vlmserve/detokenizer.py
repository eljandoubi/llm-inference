from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str: ...


class Detokenizer:
    """Turns a stream of token ids into text without decoding the whole thing again.

    Only a small window of recent tokens is decoded each time. The window is needed
    because tokenizers decide things like a leading space from the tokens before.
    A character that is still incomplete (a multi-byte one split over tokens) is
    held back until the rest arrives.
    """

    def __init__(self, tokenizer: Tokenizer, skip_special_tokens: bool = True) -> None:
        self._tokenizer = tokenizer
        self._skip = skip_special_tokens
        self._ids: list[int] = []
        self._prefix = 0  # the window starts here
        self._read = 0  # tokens before this one already became text

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        return self._decode_new(final=False)

    def flush(self) -> str:
        """What is left at the end of the stream, incomplete characters included."""
        return self._decode_new(final=True)

    def _decode_new(self, final: bool) -> str:
        old = self._tokenizer.decode(self._ids[self._prefix : self._read], self._skip)
        new = self._tokenizer.decode(self._ids[self._prefix :], self._skip)
        if len(new) <= len(old) or (new.endswith("\ufffd") and not final):
            return ""
        self._prefix, self._read = self._read, len(self._ids)
        return new[len(old) :]


class StopFilter:
    """Lets text through only once it can no longer grow into a stop string."""

    def __init__(self, stop: Sequence[str]) -> None:
        self._stop = tuple(stop)
        self._pending = ""  # text that might still be the start of a stop string

    def push(self, text: str) -> tuple[str, bool]:
        """Returns the text that is safe to send, and whether a stop string was hit.

        On a hit the text stops right before the stop string.
        """
        self._pending += text
        hits = [i for s in self._stop if (i := self._pending.find(s)) != -1]
        if hits:
            safe = self._pending[: min(hits)]
            self._pending = ""
            return safe, True
        held = self._partial_stop_length()
        cut = len(self._pending) - held
        safe, self._pending = self._pending[:cut], self._pending[cut:]
        return safe, False

    def flush(self) -> str:
        """The held back text, for a stream that ended without hitting a stop."""
        rest, self._pending = self._pending, ""
        return rest

    def _partial_stop_length(self) -> int:
        """Longest tail of the pending text that is the start of some stop string."""
        longest = 0
        for stop in self._stop:
            for k in range(min(len(stop) - 1, len(self._pending)), longest, -1):
                if self._pending.endswith(stop[:k]):
                    longest = k
                    break
        return longest

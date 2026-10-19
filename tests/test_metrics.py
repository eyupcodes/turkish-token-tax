import pytest

from turkish_token_tax.adapters import Encoded
from turkish_token_tax.metrics import compare, language_metrics


class FakeTokenizer:
    def encode(self, text):
        return Encoded(len(text), text.count("?"), text.count("!"))


def test_denominators_and_fragmentation():
    result = compare([dict(tr="ab cd", en="abc"), dict(tr="ef", en="d")], FakeTokenizer(), 100)
    assert result["tr_en_token_ratio"] == 7 / 4
    assert result["tr"]["tokens_per_word"] == 7 / 3
    assert result["tr"]["characters_per_token"] == 1
    assert result["tr"]["isolated_word_pieces"] == {"2": 3}
    assert result["tr"]["estimated_characters_per_context"] == 100


def test_diagnostic_rates_and_unsupported():
    result = language_metrics(["?!a"], FakeTokenizer(), 10)
    assert result["unknown_rate"] == result["byte_fallback_rate"] == 1 / 3

    class Unsupported:
        def encode(self, text):
            return Encoded(1, None, None)

    result = language_metrics(["a", "b"], Unsupported(), 10)
    assert result["unknown_rate"] is None
    assert result["byte_fallback_rate"] is None


@pytest.mark.parametrize("texts,budget", [([], 10), (["hi"], 0), ([" "], 10)])
def test_invalid_input(texts, budget):
    with pytest.raises(ValueError):
        language_metrics(texts, FakeTokenizer(), budget)


def test_zero_token_output():
    class Empty:
        def encode(self, text):
            return Encoded(0, 0, 0)

    with pytest.raises(ValueError, match="no tokens"):
        language_metrics(["hello"], Empty(), 10)

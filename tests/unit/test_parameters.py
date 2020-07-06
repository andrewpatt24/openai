import pytest
from openai.parameters import FixedEpsilon, ExponentialEpsilon, Epsilon


class TestParameters:

    def test_fixed_epsilon_returns_value(self):
        epsilon_value = 0.8
        e = FixedEpsilon(epsilon=epsilon_value)
        assert epsilon_value == e()
        assert epsilon_value == e()

    def test_exponential_epsilon_returns_value(self):
        decay_rate = 0.7
        e = ExponentialEpsilon(decay_rate=decay_rate)
        assert e() == 1.0
        assert round(e(), 4) == 0.5016

    def test_epsilon_return_correct_values_fixed(self):
        epsilon_value = 0.8
        e = Epsilon(epsilon=epsilon_value)
        assert epsilon_value == e()
        assert epsilon_value == e()

    def test_epsilon_return_correct_values_exponential(self):
        decay_rate = 0.7
        e = Epsilon(decay_rate=decay_rate)
        assert e() == 1.0
        assert round(e(), 4) == 0.5016

    def test_assert_defining_both_decay_and_epsilon(self):
        epsilon_value = 0.8
        decay_rate = 0.7
        with pytest.raises(AssertionError) as e:
            Epsilon(decay_rate=decay_rate, epsilon=epsilon_value)
        assert str(e.value) == 'Both decay_rate and epsilon are specified. Pick one.'

    def test_assert_defining_neither_decay_and_epsilon(self):
        with pytest.raises(AssertionError) as e:
            Epsilon(decay_rate=None, epsilon=None)
        assert str(e.value) == 'Please specify either decay_rate or epsilon'

    def test_assert_no_specifying(self):
        with pytest.raises(AssertionError) as e:
            Epsilon()
        assert str(e.value) == 'Please specify either decay_rate or epsilon'



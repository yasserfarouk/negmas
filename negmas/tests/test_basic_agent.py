import pytest

from negmas import Negotiator


def test_basic_agent_has_good_name():
    class MyAgent(Negotiator):
        def ufun(self, negotiation_id):
            return None

        def isin(self, negotiation_id) -> bool:
            return False

        def leave(self, negotiation) -> bool:
            return False

        def enter(self, negotiation, ufun) -> bool:
            return False

    x = MyAgent()

    assert x.isin(None) is False
    assert x.leave(None) is False
    assert x.enter(None, None) is False


if __name__ == '__main__':
    pytest.main(args=[__file__])

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from counterfactual_regret import KuhnPoker, LeducHoldem, build_tree  # noqa: E402


@pytest.fixture(scope="session")
def kuhn():
    return build_tree(KuhnPoker())


@pytest.fixture(scope="session")
def leduc():
    return build_tree(LeducHoldem())

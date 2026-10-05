import pytest

from splitcheck.constants import DEFAULT_SEED
from splitcheck.pipelines import run_lab


@pytest.fixture(scope="session")
def seeded_lab():
    """One seed-42 lab shared by the score tests. Reproducibility tests call run_lab themselves."""
    return run_lab(DEFAULT_SEED)

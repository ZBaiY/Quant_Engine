import os
import random

import pytest

@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")

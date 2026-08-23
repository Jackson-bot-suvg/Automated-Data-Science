from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def unique_thread_id() -> str:
    return f"test-thread-{uuid.uuid4().hex}"

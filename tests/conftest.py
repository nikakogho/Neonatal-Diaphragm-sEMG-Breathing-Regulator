"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def repo_root():
    """Return the repository root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def torch_device():
    """Provide torch device for tests."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

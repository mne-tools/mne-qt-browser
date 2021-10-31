import pytest

from mne.viz import use_browser_backend
from mne.conftest import *  # noqa: F401, F403


@pytest.fixture
def browser_backend(garbage_collect):
    """Parametrizes the name of the browser backend."""
    with use_browser_backend('pyqtgraph') as backend:
        yield backend

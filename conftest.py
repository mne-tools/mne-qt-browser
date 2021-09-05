import pytest

from mne.conftest import *

@pytest.fixture(params=['pyqtgraph'])
def browse_backend(request, garbage_collect):
    """Parametrizes the name of the browser backend."""
    with use_browser_backend(request.param) as backend:
        yield backend
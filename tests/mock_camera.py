import numpy as np
import pytest
from unittest.mock import Mock, AsyncMock

class AsyncMockVideoCapture:
    def __init__(self, source):
        self.source = source
        self._is_opened = True
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    async def isOpened(self):
        return self._is_opened
    
    async def read(self):
        return True, self.frame
    
    async def release(self):
        self._is_opened = False

@pytest.fixture
def mock_cv2(monkeypatch):
    """Mock cv2 for testing"""
    mock_cv2 = AsyncMock()
    mock_cv2.VideoCapture = AsyncMockVideoCapture
    monkeypatch.setattr('cv2.VideoCapture', mock_cv2.VideoCapture)
    return mock_cv2 
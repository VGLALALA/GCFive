import numpy as np
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ultralytics if not available
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Import the modules to test
from image_processing.ballDetection import detect_golfballs, get_detected_balls_info
from spin.GolfBall import GolfBall


class MockTensor:
    """Mock tensor that behaves like PyTorch tensor"""
    def __init__(self, data):
        self.data = np.array(data)
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
    def astype(self, dtype):
        return self.data.astype(dtype)
    
    def flatten(self):
        return self.data.flatten()
    
    def item(self):
        if self.data.size == 1:
            return self.data.item()
        return self.data[0].item()


class MockBox:
    """Mock individual box"""
    def __init__(self, class_id, box_coords):
        self.cls = MockTensor([class_id])
        self.xyxy = MockTensor([box_coords])


class MockBoxes:
    """Mock boxes that behaves like YOLO boxes"""
    def __init__(self, class_ids=None, boxes=None):
        if class_ids is None:
            class_ids = [0]  # Default to golf ball class
        if boxes is None:
            boxes = [[10, 10, 30, 30]]  # Default bounding box
        
        self.boxes = []
        for class_id, box in zip(class_ids, boxes):
            self.boxes.append(MockBox(class_id, box))
    
    def __len__(self):
        return len(self.boxes)
    
    def __iter__(self):
        return iter(self.boxes)


class MockResult:
    """Mock YOLO result"""
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else MockBoxes()


class MockYOLO:
    """Mock YOLO model for testing"""
    def __init__(self, model_path):
        self.model_path = model_path
    
    def predict(self, source, conf=0.25, imgsz=640, verbose=False):
        return [MockResult()]


@pytest.fixture
def mock_yolo(monkeypatch):
    """Mock YOLO model"""
    # Always mock YOLO for consistent testing
    monkeypatch.setattr("image_processing.ballDetection.YOLO", MockYOLO)
    monkeypatch.setattr("image_processing.ballDetection._HAS_ULTRALYTICS", True)
    monkeypatch.setattr("image_processing.ballDetection.model", MockYOLO("fake_model.pt"))


def test_detect_golfballs_with_mock(mock_yolo):
    """Test detect_golfballs with mock YOLO"""
    # Create a test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test detection
    results = detect_golfballs(image, conf=0.25, imgsz=640, display=False)
    
    assert len(results) == 1
    assert len(results[0]) == 3  # (x_center, y_center, radius)
    assert results[0][0] == 20  # x_center
    assert results[0][1] == 20  # y_center
    assert results[0][2] == 10  # radius


def test_detect_golfballs_no_results(mock_yolo, monkeypatch):
    """Test detect_golfballs when no detections"""
    # Mock empty results
    class MockEmptyResult:
        def __init__(self):
            self.boxes = None
    
    def mock_predict(*args, **kwargs):
        return [MockEmptyResult()]
    
    monkeypatch.setattr("image_processing.ballDetection.model.predict", mock_predict)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detect_golfballs(image, display=False)
    
    assert results == []


def test_detect_golfballs_wrong_class(mock_yolo, monkeypatch):
    """Test detect_golfballs with wrong class ID"""
    # Mock results with wrong class
    wrong_boxes = MockBoxes(class_ids=[1], boxes=[[10, 10, 30, 30]])
    
    def mock_predict(*args, **kwargs):
        return [MockResult(boxes=wrong_boxes)]
    
    monkeypatch.setattr("image_processing.ballDetection.model.predict", mock_predict)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detect_golfballs(image, display=False)
    
    assert results == []


def test_get_detected_balls_info_with_mock(mock_yolo):
    """Test get_detected_balls_info with mock YOLO"""
    # Create a test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test detection
    result = get_detected_balls_info(image, conf=0.25, imgsz=640)
    
    assert result is not None
    assert isinstance(result, GolfBall)
    assert result.x == 20
    assert result.y == 20
    assert result.measured_radius_pixels == 10


def test_get_detected_balls_info_grayscale(mock_yolo):
    """Test get_detected_balls_info with grayscale image"""
    # Create a grayscale test image
    image = np.zeros((100, 100), dtype=np.uint8)
    
    # Test detection
    result = get_detected_balls_info(image, conf=0.25, imgsz=640)
    
    assert result is not None
    assert isinstance(result, GolfBall)


def test_get_detected_balls_info_no_detection(mock_yolo, monkeypatch):
    """Test get_detected_balls_info when no detection"""
    # Mock empty results
    def mock_predict(*args, **kwargs):
        return []
    
    monkeypatch.setattr("image_processing.ballDetection.model.predict", mock_predict)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = get_detected_balls_info(image)
    
    assert result is None


def test_get_detected_balls_info_no_model(monkeypatch):
    """Test get_detected_balls_info when YOLO model is not available"""
    monkeypatch.setattr("image_processing.ballDetection.model", None)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = get_detected_balls_info(image)
    
    assert result is None


def test_detect_golfballs_no_ultralytics(monkeypatch):
    """Test detect_golfballs when ultralytics is not available"""
    monkeypatch.setattr("image_processing.ballDetection._HAS_ULTRALYTICS", False)
    monkeypatch.setattr("image_processing.ballDetection.model", None)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with pytest.raises(ImportError, match="ultralytics YOLO model not available"):
        detect_golfballs(image)


def test_detect_golfballs_multiple_balls(mock_yolo, monkeypatch):
    """Test detect_golfballs with multiple balls"""
    # Mock multiple detections
    multiple_boxes = MockBoxes(
        class_ids=[0, 0],  # Two golf balls
        boxes=[[10, 10, 30, 30], [50, 50, 70, 70]]
    )
    
    def mock_predict(*args, **kwargs):
        return [MockResult(boxes=multiple_boxes)]
    
    monkeypatch.setattr("image_processing.ballDetection.model.predict", mock_predict)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detect_golfballs(image, display=False)
    
    assert len(results) == 2
    # Should be sorted left to right
    assert results[0][0] < results[1][0]  # First ball left of second ball


def test_detect_golfballs_empty_boxes(mock_yolo, monkeypatch):
    """Test detect_golfballs with empty boxes"""
    # Mock empty boxes
    empty_boxes = MockBoxes(class_ids=[], boxes=[])
    
    def mock_predict(*args, **kwargs):
        return [MockResult(boxes=empty_boxes)]
    
    monkeypatch.setattr("image_processing.ballDetection.model.predict", mock_predict)
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detect_golfballs(image, display=False)
    
    assert results == []

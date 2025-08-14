import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "utility.backspin_client",
        "spin.spinAxis",
        "trajectory_simulation.ball",
        "trajectory_simulation.vector",
    ],
)
def test_basic_imports(module):
    if module == "utility.backspin_client":
        pytest.importorskip(
            "requests", reason="requests not installed", exc_type=ImportError
        )
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    ["image_processing.FormatImage", "image_processing.Convert_GrayScale"],
)
def test_cv2_dependent_imports(module):
    pytest.importorskip("cv2", reason="OpenCV not installed", exc_type=ImportError)
    importlib.import_module(module)

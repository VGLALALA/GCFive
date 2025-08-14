# Image Processing Module

Documentation for files in `image_processing/`.

- `ApplyGaborFilter.py`: Creates and applies Gabor filters for feature enhancement.
- `CamBalldistancePred.py`: Predicts camera-to-ball distance using calibration data and detections.
- `Convert_Canny.py`: Converts images to edges using the Canny operator.
- `Convert_GrayScale.py`: Utility to convert images to grayscale.
- `FormatImage.py`: Detects a ball and returns it as a `GolfBall` instance.
- `ImageCompressor.py`: Compresses grayscale images by a scale factor.
- `ImgComparsionOp.py`: Compares a target image against candidate rotation images.
- `IsolateCode.py`: Crops an image around a detected golf ball.
- `MaskAreaOutsideBall.py`: Masks pixels outside the ball's region to a constant value.
- `Normalization.py`: Contains experimental code for normalizing rotation angles.
- `ROI.py`: Selects the best region of interest for a golf ball in an image.
- `RemoveReflection.py`: Filters out bright reflections in a grayscale image.
- `SimilarityCalculation.py`: Computes similarity between two frames using grayscale differences.
- `ballDetection.py`: Runs YOLO to detect golf balls in frames.
- `ballDetectionyolo.py`: Alternate YOLO-based ball detector with visualization options.
- `ballSpeedCalculation.py`: Estimates ball speed from two frames and time delta.
- `ballinZoneCheck.py`: Geometry helpers to check if the ball lies within a polygonal zone.
- `get2Dcoord.py`: Estimates ball position in 2D space using calibration data.
- `launchAngleCalculation.py`: Calculates launch angle from two detections.
- `matchBallSize.py`: Pads or crops images so both contain similarly sized balls.
- `movementDetection.py`: Determines if the ball has moved between frames.
- `__init__.py`: Package marker for image processing utilities.

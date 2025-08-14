# Spin Module

Documentation for files in `spin/`.

- `CompareCandidateAngleImage.py`: Compares rendered candidate images against a target to score rotation angles.
- `CompareRotationImage.py`: Computes pixel differences between two rotation images.
- `GenerateRotationCandidate.py`: Builds a grid of rotation candidates by projecting and rotating a base image.
- `GetBallRotation.py`: High-level pipeline to estimate the ball's 3D rotation between frames.
- `GetBallRotation.spec`: PyInstaller spec describing how to bundle the rotation tool.
- `GetLaunchAngle.py`: Estimates the ball's launch angle from two frames.
- `GetRotatedImage.py`: Produces a rotated view of the ball image using projection and unprojection.
- `GolfBall.py`: Data class representing ball geometry and camera parameters.
- `GradientDescent.py`: Refines rotation estimates using numerical optimization.
- `HyperParameter.json`: Default hyperparameters for the rotation search process.
- `Project2dImageTo3dBall.py`: Projects a 2D ball image onto a 3D hemisphere and rotates it.
- `ProjectOp.py`: Helper class that performs low-level projection operations.
- `RotationCandidate.py`: Data class storing a candidate image and its rotation angles.
- `RotationSearch.py`: Defines coarse search ranges for rotation axes.
- `RotationSearchSpace.py`: Structure describing rotation search bounds for each axis.
- `Unproject3Dto2D.py`: Converts a 3D projected ball image back to 2D.
- `Vector2RPM.py`: Converts a rotation vector into backspin and sidespin in RPM.
- `spinAxis.py`: Calculates spin axis direction from backspin and sidespin values.
- `__init__.py`: Package marker for spin utilities.

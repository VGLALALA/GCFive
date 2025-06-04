This folder contains a translation of the Godot scripts from the JaySimG project.
The original project can be found at: https://github.com/jhauck2/JaySimG

The scripts here provide a Python approximation of the `ball.gd`, `ball_trail.gd`,
and `range.gd` Godot scripts. They are not a drop-in replacement for the Godot
implementation but aim to replicate the core logic so that the physics model can
be run in a plain Python environment.

## Visualizing a sample shot
Run `python -m JaySimG_translation.visualize_flight` to simulate a default shot
and save a `trajectory.png` image along with printed carry distance, time of
flight, and apex height.

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go


def visualize_golf_ball(
    positions,
    speed_mps,
    launch_angle,
    backspin_rpm,
    side_spin_rpm,
    spin_axis,
    carry,
    total,
    apex,
    time_of_flight,
    descending_angle,
):
    positions = np.array(positions)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # 2D Plot: Down the Line View
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, marker="o")
    ax1.set_xlabel("Downrange Distance (m)")
    ax1.set_ylabel("Height (m)")
    ax1.set_title("Down the Line View")
    fig1.tight_layout()

    # 2D Plot: Side View
    fig2, ax2 = plt.subplots()
    ax2.plot(z, y, marker="o", linestyle="--")
    ax2.set_xlabel("Lateral Dispersion (m)")
    ax2.set_ylabel("Height (m)")
    ax2.set_title("Side View")
    fig2.tight_layout()

    # Interactive 3D Plot
    fig3 = go.Figure(
        data=[go.Scatter3d(x=x, y=z, z=y, mode="lines+markers", marker=dict(size=3))],
        layout=go.Layout(
            title="3D Trajectory",
            scene=dict(
                xaxis=dict(title="Downrange (m)"),
                yaxis=dict(title="Lateral (m)"),
                zaxis=dict(title="Height (m)"),
            ),
        ),
    )

    return fig1, fig2, fig3


# Define Gradio interface
interface = gr.Interface(
    fn=visualize_golf_ball,
    inputs=[
        gr.Dataframe(
            headers=["X (m)", "Y (m)", "Z (m)"],
            label="Positions (3D array of shape [n, 3])",
        ),
        gr.Number(label="Speed (m/s)"),
        gr.Number(label="Launch Angle (°)"),
        gr.Number(label="Backspin (RPM)"),
        gr.Number(label="Side Spin (RPM)"),
        gr.Number(label="Spin Axis (°)"),
        gr.Number(label="Carry Distance (m)"),
        gr.Number(label="Total Distance (m)"),
        gr.Number(label="Apex Height (m)"),
        gr.Number(label="Time of Flight (s)"),
        gr.Number(label="Descending Angle (°)"),
    ],
    outputs=[
        gr.Plot(label="Down the Line View"),
        gr.Plot(label="Side View"),
        gr.Plot(label="Interactive 3D Trajectory"),
    ],
    title="Golf Ball Trajectory Visualizer",
    description="Enter your ball‑flight data to generate 2D and interactive 3D plots.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

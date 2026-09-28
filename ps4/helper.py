import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from itertools import cycle

def produce_animation(
    history,
    stop_names,
    stop_positions,
    train_names,
    track_length,
    show=True,
    interval=250,
    radius=10,
):
    """
    Produces an animation of trains moving counter-clockwise around a circular track.
    Position 0 is at the middle right (3 o’clock).
    """
    num_frames = len(history)
    num_trains = len(history[0])

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-radius - 2, radius + 2)
    ax.set_ylim(-radius - 2, radius + 2)
    ax.axis("off")

    # Draw circular track
    track = plt.Circle((0, 0), radius, fill=False, color="black", linewidth=2)
    ax.add_artist(track)

    # --- Mapping: 0 = 3 o’clock, increasing counter-clockwise ---
    def convert_to_radians(loc):
        return 2 * np.pi * (loc % track_length) / track_length

    # --- Draw tick marks for stops ---
    tick_length = 0.4
    for mile in stop_positions:
        angle = convert_to_radians(mile)
        x_inner, y_inner = radius * np.cos(angle), radius * np.sin(angle)
        x_outer, y_outer = (radius + tick_length) * np.cos(angle), (radius + tick_length) * np.sin(angle)
        ax.plot([x_inner, x_outer], [y_inner, y_outer], color="black", linewidth=2)

    # --- Draw stop labels ---
    label_radius = radius + tick_length + 0.9
    for stop_name, mile in zip(stop_names, stop_positions):
        angle = convert_to_radians(mile)
        x, y = label_radius * np.cos(angle), label_radius * np.sin(angle)
        ax.text(x, y, stop_name, ha="center", va="center", fontsize=9, fontweight="bold")

    # --- Train markers + labels ---
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    trains, train_labels = [], []
    for name in train_names[:num_trains]:
        color = next(color_cycle)
        train, = ax.plot([], [], "o", color=color, markersize=10)
        trains.append(train)
        label = ax.text(0, 0, name, ha="center", va="bottom", fontsize=8)
        train_labels.append(label)

    # --- Time step text ---
    timestep_text = ax.text(
        radius, -radius - 0.5, "", ha="right", va="bottom", fontsize=10, color="black"
    )

    # --- Update animation frame ---
    def update(frame):
        for i, (train, label) in enumerate(zip(trains, train_labels)):
            loc = history[frame][i]
            angle = convert_to_radians(loc)
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            train.set_data([x], [y])

            # Label slightly offset outward
            offset = 0.5
            label_x = (radius + offset) * np.cos(angle)
            label_y = (radius + offset) * np.sin(angle)
            label.set_position((label_x, label_y))

        timestep_text.set_text(f"t = {frame}")
        return trains + train_labels + [timestep_text]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    if show:
        plt.show()
    plt.close(fig)
    return ani
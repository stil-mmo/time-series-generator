import matplotlib.pyplot as plt
import numpy as np

from tsg.process.process_list import ProcessList
from tsg.time_series import TimeSeries


def save_data(
    time_series: TimeSeries, process_list: ProcessList, log_path: str, ts_number: int
):
    with open(log_path, "a") as log_file:
        log_file.write("\n")
        log_file.write(f"TS {ts_number}\n")
        last_steps = 0
        for i in range(len(time_series.metadata)):
            current_process_data = time_series.metadata[i]
            current_process = process_list.get_processes([current_process_data[0]])[0]
            log_file.write(f"Process {i}: {current_process_data[0]}\n")
            sample = current_process_data[1]
            log_file.write(
                f"Starts at {last_steps}, ends at {sample[0] + last_steps - 1}, all={sample[0]}\n"
            )
            log_file.write(str(current_process.get_info(data=sample)))
            log_file.write("\n")
            last_steps = sample[0] + last_steps


def save_plot(
    coordinates, clusters, border_values, shift, time_series_array, save_plot_path=None
):
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Generated time series with parameters clustering")

    ax = fig.add_subplot(1, 2, 1)
    colors = ["blue", "red", "green"]
    for i in range(5):
        time_series = time_series_array[i]
        ax.text(100, time_series[-1], f"TS {i}")
        ax.plot(time_series, color=colors[clusters[i]])
    ax.grid(True)
    ax.set_xlabel("time")
    ax.set_ylabel("values")

    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection="3d")

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    x += shift
    y += shift
    z += shift

    ax.plot_wireframe(x, y, z, color="grey", alpha=0.3)
    for i in range(5):
        ax.scatter(
            coordinates[i][0],
            coordinates[i][1],
            coordinates[i][2],
            color=colors[clusters[i]],
        )
    ax.set_zlim(border_values[0], border_values[1])
    if save_plot_path is None:
        plt.show()
    else:
        plt.savefig(save_plot_path)

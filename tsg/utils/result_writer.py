import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from tsg.time_series import TimeSeries
from tsg.utils.typing import NDArrayFloat64T


def save_parameters(ts_list: list[TimeSeries], json_path: str) -> None:
    with open(json_path, "w") as json_file:
        json_data = get_json_data(ts_list)
        json_file.write(json.dumps(json_data, indent=4))


def load_parameters(json_path: str) -> dict:
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data


def save_values(array: NDArrayFloat64T, csv_path: str) -> None:
    with open(csv_path, "w") as csv_file:
        np.savetxt(csv_file, array)


def load_values(csv_path: str) -> NDArrayFloat64T:
    return np.genfromtxt(csv_path)


def get_json_data(ts_list: list[TimeSeries]) -> dict:
    json_data = {}
    for i in range(len(ts_list)):
        last_steps = 0
        time_series = ts_list[i]
        ts_json_data = []
        ts_name = f"ts_{i + 1}"
        for j in range(len(time_series.metadata)):
            current_metadata = time_series.metadata[j]
            current_process_data = current_metadata[1]
            current_process_name = current_metadata[0]
            process_json_data = {
                "name": current_process_name,
                "start": last_steps,
                "end": current_process_data[0] + last_steps - 1,
                "params": current_process_data[1].tolist(),
            }
            ts_json_data.append(process_json_data)
            last_steps = current_process_data[0] + last_steps
        json_data[ts_name] = ts_json_data
    return json_data


def save_plot(
    coordinates: NDArrayFloat64T,
    clusters: NDArrayFloat64T,
    border_values: tuple[float, float],
    shift: float,
    time_series_array: NDArrayFloat64T,
    save_plot_path=None,
):
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Generated time series with parameters clustering")

    ax_graph = fig.add_subplot(1, 2, 1)
    colors = ["blue", "red", "green"]
    for i in range(5):
        time_series = time_series_array[i]
        ax_graph.text(100, time_series[-1], f"TS {i}")
        ax_graph.plot(time_series, color=colors[clusters[i]])
    ax_graph.grid(True)
    ax_graph.set_xlabel("time")
    ax_graph.set_ylabel("values")

    ax_sphere: Axes3D = fig.add_subplot(1, 2, 2, projection="3d")

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    x += shift
    y += shift
    z += shift

    ax_sphere.plot_wireframe(x, y, z, color="grey", alpha=0.3)
    for i in range(5):
        ax_sphere.scatter(
            coordinates[i][0],
            coordinates[i][1],
            coordinates[i][2],
            color=colors[clusters[i]],
        )
    ax_sphere.set_zlim(border_values[0], border_values[1])
    if save_plot_path is None:
        plt.show()
    else:
        plt.savefig(save_plot_path)

import os

import hydra
import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.sampling.point_clustering import cluster_points
from tsg.sampling.point_sampling import sample_points
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series_generator import TimeSeriesGenerator
from tsg.utils.result_writer import save_parameters, save_plot, save_values


@hydra.main(version_base="1.2", config_path="..", config_name="config")
def main(cfg: DictConfig):
    if cfg.generation.sample_points:
        coordinates, border_values, shift = sample_points(cfg.generation.ts_number)
        clusters = cluster_points(coordinates, cfg.clustering.clusters)
        print(coordinates)
        print(f"Clusters: {clusters}")
        print(shift)
        linspace_info = LinspaceInfo(
            start=border_values[0],
            stop=border_values[1],
            parts=cfg.linspace_info.linspace_parts,
            center_shift=cfg.linspace_info.center_shift,
            step_coeff=cfg.linspace_info.step_coeff,
            use_k=cfg.linspace_info.use_k,
        )
        print(
            linspace_info.start,
            linspace_info.stop,
            linspace_info.step,
        )
        storage = SchedulerStorage(
            cfg=cfg,
            linspace_info=linspace_info,
            points=coordinates,
            clusters=clusters,
        )
    else:
        coordinates = np.array([])
        border_values = (np.float64(0.0), np.float64(0.0))
        shift = np.float64(0.0)
        clusters = np.array([])
        linspace_info = LinspaceInfo(
            start=cfg.linspace_info.linspace_borders[0],
            stop=cfg.linspace_info.linspace_borders[1],
            parts=cfg.linspace_info.linspace_parts,
            center_shift=cfg.linspace_info.center_shift,
            step_coeff=cfg.linspace_info.step_coeff,
            use_k=cfg.linspace_info.use_k,
        )
        storage = None
    ts_generator = TimeSeriesGenerator(
        cfg=cfg,
        linspace_info=linspace_info,
        scheduler_storage=storage,
    )
    time_series_array, time_series_list = ts_generator.generate_all()
    folder = cfg.generation.save_data_folder
    os.makedirs(folder, exist_ok=True)
    save_data_path = os.path.join(folder, cfg.generation.json_name)
    save_values_path = os.path.join(folder, cfg.generation.csv_name)
    save_plot_path = os.path.join(folder, cfg.generation.plot_name)
    save_parameters(time_series_list, save_data_path)
    save_values(time_series_array, save_values_path)
    if cfg.generation.sample_points:
        save_plot(
            coordinates,
            clusters,
            border_values,
            shift,
            time_series_array,
            save_plot_path,
        )


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_storage import ProcessStorage
from tsg.sampling.point_clustering import cluster_points
from tsg.sampling.point_sampling import sample_points
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series_generator import TimeSeriesGenerator
from tsg.utils.result_writer import save_parameters, save_plot, save_values
from tsg.utils.utils import get_config_path


@hydra.main(
    version_base="1.2",
    config_path=get_config_path(path=Path(__file__)),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    generation_method_name = cfg.generation.generation_method
    if cfg.parameters_generation_method.get(generation_method_name) is not None:
        method_partial = hydra.utils.instantiate(
            cfg.parameters_generation_method[generation_method_name], _partial_=True
        )
    else:
        method_partial = hydra.utils.instantiate(
            cfg.parameters_generation_method["random_method"], _partial_=True
        )
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
        generation_method = method_partial(linspace_info=linspace_info)
        process_storage = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=linspace_info,
            generation_method=generation_method,
        )
        storage = SchedulerStorage(
            num_steps=cfg.generation.ts_size,
            cfg_scheduler=cfg.scheduler,
            linspace_info=linspace_info,
            process_storage=process_storage,
            source_points=coordinates,
            clusters=clusters,
        )
    else:
        coordinates = np.array([])
        border_values = (0.0, 0.0)
        shift = 0.0
        clusters = np.array([])
        linspace_info = LinspaceInfo(
            start=cfg.linspace_info.linspace_borders[0],
            stop=cfg.linspace_info.linspace_borders[1],
            parts=cfg.linspace_info.linspace_parts,
            center_shift=cfg.linspace_info.center_shift,
            step_coeff=cfg.linspace_info.step_coeff,
            use_k=cfg.linspace_info.use_k,
        )
        generation_method = method_partial(linspace_info=linspace_info)
        process_storage = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=linspace_info,
            generation_method=generation_method,
        )
        storage = None
    ts_generator = TimeSeriesGenerator(
        cfg=cfg,
        linspace_info=linspace_info,
        process_storage=process_storage,
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

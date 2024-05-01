import os
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.process_storage import ProcessStorage
from tsg.sampling.point_clustering import cluster_points
from tsg.sampling.point_sampling import sample_points
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series import TimeSeries
from tsg.time_series_generator import TimeSeriesGenerator
from tsg.utils.result_writer import save_parameters, save_plot, save_values
from tsg.utils.typing import NDArrayFloat64T
from tsg.utils.utils import get_config_path


@hydra.main(
    version_base="1.2",
    config_path=get_config_path(path=Path(__file__)),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    method_partial = get_generation_method(cfg)

    if cfg.generation.sample_points:
        coordinates, clusters, border_values, shift = generate_source_data(cfg)
        plot_data = [coordinates, clusters, border_values, shift]
        linspace_info = get_linspace_info(
            cfg=cfg, start=border_values[0], stop=border_values[1]
        )
        generation_method = method_partial(linspace_info=linspace_info)
        process_storage = get_process_storage(cfg, linspace_info, generation_method)
        scheduler_storage = SchedulerStorage(
            num_steps=cfg.generation.ts_size,
            cfg_scheduler=cfg.scheduler,
            linspace_info=linspace_info,
            process_storage=process_storage,
            source_points=coordinates,
            clusters=clusters,
        )
    else:
        plot_data = None
        linspace_info = get_linspace_info(
            cfg=cfg,
            start=cfg.linspace_info.linspace_borders[0],
            stop=cfg.linspace_info.linspace_borders[1],
        )
        generation_method = method_partial(linspace_info=linspace_info)
        process_storage = get_process_storage(cfg, linspace_info, generation_method)
        scheduler_storage = None

    ts_generator = TimeSeriesGenerator(
        cfg=cfg,
        linspace_info=linspace_info,
        process_storage=process_storage,
        scheduler_storage=scheduler_storage,
    )
    time_series_array, time_series_list = ts_generator.generate_all()
    save_results(
        cfg=cfg,
        ts_array=time_series_array,
        ts_list=time_series_list,
        plot_data=plot_data,
    )


def get_generation_method(cfg: DictConfig) -> Callable:
    generation_method_name = cfg.generation.generation_method
    if cfg.parameters_generation_method.get(generation_method_name) is not None:
        method_partial = hydra.utils.instantiate(
            cfg.parameters_generation_method[generation_method_name], _partial_=True
        )
    else:
        method_partial = hydra.utils.instantiate(
            cfg.parameters_generation_method["random_method"], _partial_=True
        )
    return method_partial


def generate_source_data(
    cfg: DictConfig,
) -> tuple[NDArrayFloat64T, NDArray[np.int_], tuple[float, float], float]:
    coordinates, border_values, shift = sample_points(cfg.generation.ts_number)
    clusters = cluster_points(coordinates, cfg.clustering.clusters)
    return coordinates, clusters, border_values, shift


def get_linspace_info(cfg: DictConfig, start: float, stop: float) -> LinspaceInfo:
    return LinspaceInfo(
        start=start,
        stop=stop,
        parts=cfg.linspace_info.linspace_parts,
        center_shift=cfg.linspace_info.center_shift,
        step_coeff=cfg.linspace_info.step_coeff,
        use_k=cfg.linspace_info.use_k,
    )


def get_process_storage(
    cfg: DictConfig,
    linspace_info: LinspaceInfo,
    generation_method: ParametersGenerationMethod,
) -> ProcessStorage:
    return ProcessStorage(
        process_list=cfg.scheduler.process_list,
        cfg_process=cfg.process,
        linspace_info=linspace_info,
        generation_method=generation_method,
    )


def save_results(
    cfg: DictConfig,
    ts_array: NDArrayFloat64T,
    ts_list: list[TimeSeries],
    plot_data: list | None = None,
) -> None:
    folder = cfg.generation.save_data_folder
    os.makedirs(folder, exist_ok=True)
    save_data_path = os.path.join(folder, cfg.generation.json_name)
    save_values_path = os.path.join(folder, cfg.generation.csv_name)
    save_plot_path = os.path.join(folder, cfg.generation.plot_name)
    save_parameters(ts_list, save_data_path)
    save_values(ts_array, save_values_path)
    if plot_data is not None:
        plot_data.append(ts_array)
        plot_data.append(save_plot_path)
        save_plot(*plot_data)


if __name__ == "__main__":
    main()

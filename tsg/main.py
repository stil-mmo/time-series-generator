import os

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_list import ProcessList
from tsg.sampling.point_clustering import cluster_points
from tsg.sampling.point_sampling import sample_points
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series_generator import TimeSeriesGenerator
from tsg.utils.result_writer import save_parameters, save_plot, save_values

if __name__ == "__main__":
    coordinates, border_values, shift = sample_points(5)
    clusters = cluster_points(coordinates, 2)
    print(coordinates)
    print(f"Clusters: {clusters}")
    print(shift)
    test_generator_linspace = LinspaceInfo(
        start=border_values[0], stop=border_values[1], parts=100
    )
    print(
        test_generator_linspace.start,
        test_generator_linspace.stop,
        test_generator_linspace.step,
    )
    storage = SchedulerStorage(
        num_steps=100,
        generator_linspace=test_generator_linspace,
        points=coordinates,
        clusters=clusters,
    )
    ts_generator = TimeSeriesGenerator(
        num_time_series=coordinates.shape[0],
        num_steps=100,
        linspace_info=test_generator_linspace,
        scheduler_storage=storage,
        stable_parameters=True,
        single_schedule=False,
    )
    pl = ProcessList()
    pl.add_all_processes(test_generator_linspace)
    time_series_array, time_series_list = ts_generator.generate_all()
    os.makedirs("saved_data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    save_data_path = os.path.join("saved_data", "generation.json")
    save_values_path = os.path.join("saved_data", "values.csv")
    save_plot_path = os.path.join("plots", "plot.png")
    save_parameters(time_series_list, save_data_path)
    save_values(time_series_array, save_values_path)
    save_plot(
        coordinates, clusters, border_values, shift, time_series_array, save_plot_path
    )

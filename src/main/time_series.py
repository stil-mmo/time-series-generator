from numpy import array
from numpy.typing import NDArray


class TimeSeries:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.last_index = 0
        self.values = array([0.0 for _ in range(0, num_steps)])
        self.metadata = []

    def add_values(
        self,
        new_values: NDArray,
        new_metadata: tuple[str, tuple[int, tuple[float, ...]]],
    ) -> None:
        if self.last_index + len(new_values) > self.num_steps:
            print(
                f"Number of values to add in time series exceeds the number of steps: {len(new_values)} > {self.num_steps}"
            )
        self.values[self.last_index : self.last_index + len(new_values)] = new_values
        self.last_index += len(new_values)
        self.metadata.append(new_metadata)

    def get_values(self, start_index: int = 0, end_index: int | None = None) -> NDArray:
        if end_index is None:
            return self.values[start_index : self.last_index]
        else:
            return self.values[start_index:end_index]

    def dump_logs(self, process_list, log_path: str, ts_number: int):
        with open(log_path, "a") as log_file:
            log_file.write("\n")
            log_file.write(f"TS {ts_number}\n")
            last_steps = 0
            for i in range(len(self.metadata)):
                current_process_data = self.metadata[i]
                current_process = process_list.get_processes([current_process_data[0]])[
                    0
                ]
                log_file.write(f"Process {i}: {current_process_data[0]}\n")
                sample = current_process_data[1]
                log_file.write(
                    f"Starts at {last_steps}, ends at {sample[0] + last_steps - 1}, all={sample[0]}\n"
                )
                log_file.write(str(current_process.get_info(data=sample)))
                log_file.write("\n")
                last_steps = sample[0] + last_steps

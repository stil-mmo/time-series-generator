from schedule import Schedule


class TimeSeriesGenerator:
    def __init__(self, num_time_series, num_steps, process_id):
        self.num_time_series = num_time_series
        self.num_steps = num_steps
        self.process_id = process_id

    def generate_all(self, schedule: Schedule):
        pass

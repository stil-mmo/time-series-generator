from unittest import TestCase

from src.main.scheduler import Scheduler


class TestScheduler(TestCase):
    def test_generate_process_list(self):
        schedule = Scheduler(100)
        self.assertTrue(
            schedule.process_list.contains("white_noise")
            and schedule.process_list.contains("random_walk")
        )

    def test_generate_steps_number(self):
        num_max = 100
        num_parts = 10
        strict_steps_list = Scheduler.generate_steps_number(num_max, num_parts, True)
        self.assertEqual(len(strict_steps_list), num_parts)
        self.assertEqual(sum(strict_steps_list), num_max)
        steps_list = Scheduler.generate_steps_number(num_max, num_parts)
        self.assertTrue(len(steps_list) <= num_parts)
        self.assertEqual(sum(steps_list), num_max)

    def test_generate_process_order(self):
        schedule = Scheduler(100)
        process_order = schedule.generate_process_order()
        self.assertTrue(sum([steps for steps, _ in process_order]) == 100)
        self.assertTrue(len(process_order) <= 10)
        self.assertTrue(
            all(
                [
                    process_name in schedule.process_list.processes.keys()
                    for _, process_name in process_order
                ]
            )
        )

    def test_generate_schedule(self):
        scheduler = Scheduler(100)
        schedule = scheduler.generate_schedule(-10, 10)
        self.assertEqual(len(schedule), len(scheduler.process_order))
        steps_sum = []
        for _, process_data in schedule:
            current_steps_sum = 0
            for steps, _ in process_data:
                current_steps_sum += steps
            steps_sum.append(current_steps_sum)
        self.assertTrue(
            steps_sum[i] == scheduler.process_order[i][0] for i in range(len(steps_sum))
        )

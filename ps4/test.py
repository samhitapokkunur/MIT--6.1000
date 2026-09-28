# standard library
from functools import wraps
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from collections import OrderedDict

# local application
import pset

# DO NOT MODIFY
def case_options(points, failure, error):
    """Decorator to add points and messages to a test case"""

    def decorator(func):
        # Directly set attributes on the original function
        func.points = points
        func.failure_message = failure
        func.error_message = error

        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[-1], MagicMock):
                args = args[:-1]
            return func(*args, **kwargs)

        return wrapper

    return decorator


# DO NOT MODIFY
def testsuite_options(timeout, weight):
    """Decorator to add timeout and weight to a test suite"""

    def decorator(cls):
        # Directly set attributes on the original class
        cls.timeout = timeout
        cls.weight = weight
        return cls

    return decorator



############################################################
# Test Examples
############################################################

ACCEPTABLE_STD = 4
ACCEPTABLE_EPSILON = 1e-3

SCENARIO_1 = OrderedDict({
    "locations": [0, 1.2, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3],
    "track_length": 7,
    'speed': 12/60,
    'p': 0.2,
    'sigma': 0.1,
    'num_steps_ideal': 10,
    'num_steps_stochastic': 100,
    'num_trials': 50
})

SCENARIO_2 = OrderedDict({
    "locations": [0, 1.4, 2.5, 3.6, 4.9, 5.3, 6.4, 8.2],
    "track_length": 10,
    'speed': 15/60,
    'p': 0.5,
    'sigma': 0.8,
    'num_steps_ideal': 15,
    'num_steps_stochastic': 150,
    'num_trials': 50
})

############################################################
# Test Cases
############################################################
@testsuite_options(4, 1)
class TestPart1(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # helper function for comparing histories
    def assertHistoryEqual(self, expected: list[list], actual: list[list]):
        self.assertEqual(len(expected), len(actual), f"History lengths differ in number of steps, expected {len(expected)}, got {len(actual)}")
        for t, (exp_train_locations, actual_train_locations) in enumerate(zip(expected, actual)):
            self.assertEqual(
                len(exp_train_locations),
                len(actual_train_locations),
                f"History lengths differ in number of trains at time step {t}. Expected {len(exp_train_locations)}, got {len(actual_train_locations)}",
            )
            for i, (loc_expected, loc_actual) in enumerate(zip(exp_train_locations, actual_train_locations)):
                self.assertAlmostEqual(
                    loc_expected,
                    loc_actual,
                    delta=ACCEPTABLE_EPSILON,
                    msg=f"Train locations differ at step {t}, train {i}. Expected {loc_expected}, got {loc_actual}",
                )

    # 1.1 modular distance & will pass stop
    @case_options(
        1,
        failure="get_distance() does not return the correct modular distance",
        error="get_distance() raised an error"
    )
    def test_distance(self):
        location_1_list = [12, 3, 5]
        location_2_list = [2, 5, 5]
        track_length_list = [14, 7, 8]
        expected_distance = [4, 2, 0]

        for l1, l2, track_length, expected_distance in zip(location_1_list, location_2_list, track_length_list, expected_distance):
            actual_distance = pset.get_distance(l1, l2, track_length)
            self.assertEqual(expected_distance, actual_distance, f"got get_distance({l1}, {l2}, {track_length}) = {actual_distance}, expected {expected_distance}")

    @case_options(
        1,
        failure="will_pass_stop() does not return the correct boolean value",
        error="will_pass_stop() raised an error"
    )
    def test_will_pass_stop(self):
        location_1_list = [12, 3, 5]
        location_2_list = [2, 5, 5]
        stop_location_list = [1, 4, 6]
        track_length_list = [14, 7, 8]
        expected_output = [True, True, False]

        for l1, l2, stop_location, track_length, expected in zip(location_1_list, location_2_list, stop_location_list, track_length_list, expected_output):
            actual = pset.will_pass_stop(l1, l2, stop_location, track_length)
            self.assertEqual(expected, actual, f"got will_pass_stop({l1}, {l2}, {stop_location}, {track_length}) = {actual}, expected {expected}")

    # 1.2 simulate trains
    @case_options(
        1,
        failure="simulate_trains_ideal() does not return the correct location history",
        error="simulate_trains_ideal() raised an error"
    )
    def test_simulate_trains_ideal_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()
        expected = [[0, 1.2, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3], [0, 1.2, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3], [0, 1.2, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3], [0.2, 1.4, 2.1, 2.9000000000000004, 3.5, 4.1, 4.9, 5.5], [0.4, 1.5999999999999999, 2.3000000000000003, 3.1000000000000005, 3.7, 4.3, 5.1000000000000005, 5.7], [0.6000000000000001, 1.7999999999999998, 2.5000000000000004, 3.3, 3.9, 4.5, 5.3, 5.9], [0.8, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3, 6.1000000000000005], [1.0, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3, 6.300000000000001], [1.2, 1.9, 2.7, 3.5, 4.1, 4.7, 5.5, 6.500000000000001], [1.2, 2.1, 2.9000000000000004, 3.7, 4.3, 4.9, 5.7, 6.700000000000001], [1.2, 2.3000000000000003, 3.1000000000000005, 3.9, 4.5, 5.1000000000000005, 5.9, 6.900000000000001]]
        actual = pset.simulate_trains_ideal(locations, speed, num_steps_ideal, track_length)
        self.assertHistoryEqual(expected, actual)

    @case_options(
        1,
        failure="simulate_trains_ideal() does not return the correct location history",
        error="simulate_trains_ideal() raised an error"
    )
    def test_simulate_trains_ideal_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()
        expected = [[0, 1.4, 2.5, 3.6, 4.9, 5.3, 6.4, 8.2], [0, 1.4, 2.5, 3.6, 4.9, 5.3, 6.4, 8.2], [0, 1.4, 2.5, 3.6, 4.9, 5.3, 6.4, 8.2], [0.25, 1.65, 2.75, 3.85, 5.15, 5.55, 6.65, 8.45], [0.5, 1.9, 3.0, 4.1, 5.3, 5.8, 6.9, 8.7], [0.75, 2.15, 3.25, 4.35, 5.3, 6.05, 7.15, 8.95], [1.0, 2.4, 3.5, 4.6, 5.3, 6.3, 7.4, 9.2], [1.25, 2.5, 3.6, 4.85, 5.55, 6.4, 7.65, 9.45], [1.4, 2.5, 3.6, 4.9, 5.8, 6.4, 7.9, 9.7], [1.4, 2.5, 3.6, 4.9, 6.05, 6.4, 8.15, 9.95], [1.4, 2.75, 3.85, 4.9, 6.3, 6.65, 8.2, 0], [1.65, 3.0, 4.1, 5.15, 6.4, 6.9, 8.2, 0], [1.9, 3.25, 4.35, 5.3, 6.4, 7.15, 8.2, 0], [2.15, 3.5, 4.6, 5.3, 6.4, 7.4, 8.45, 0.25], [2.4, 3.6, 4.85, 5.3, 6.65, 7.65, 8.7, 0.5], [2.5, 3.6, 4.9, 5.55, 6.9, 7.9, 8.95, 0.75]]
        actual = pset.simulate_trains_ideal(locations, speed, num_steps_ideal, track_length)
        self.assertHistoryEqual(expected, actual)

    # 1.3 monte carlo helper functions
    @case_options(
        1,
        failure="compute_mean() does not return the correct mean",
        error="compute_mean() raised an error"
    )
    def test_compute_mean(self):
        nums = [1, 3, 8.9, 4.5, 6.7, 8.2, 1.1, 2.7, 3.1, 3.1, 0, 10.8]
        expected_mean = 4.425
        actual_mean = pset.compute_mean(nums)
        self.assertAlmostEqual(expected_mean, actual_mean, delta=ACCEPTABLE_EPSILON, msg=f"got compute_mean({nums}) = {actual_mean}, expected {expected_mean}")

    @case_options(
        1,
        failure="compute_standard_deviation() does not return the correct standard deviation",
        error="compute_standard_deviation() raised an error"
    )
    def test_compute_std(self):
        nums = [1, 3, 8.9, 4.5, 6.7, 8.2, 1.1, 2.7, 3.1, 3.1, 0, 10.8]
        expected_std = 3.3038172
        actual_std = pset.compute_standard_deviation(nums)
        self.assertAlmostEqual(expected_std, actual_std, delta=ACCEPTABLE_EPSILON, msg=f"got compute_standard_deviation({nums}) = {actual_std}, expected {expected_std}")

    # 1.4 test_monte_carlo
    @case_options(
        1,
        failure="get_intertrain_stats() does not return the correct mean and std",
        error="get_intertrain_stats() raised an error"
    )
    def test_get_intertrain_stats_1(self):
        expected_mean, expected_std = 3.75, 0.9682458365518543
        location_history = pset.simulate_trains_ideal(
            SCENARIO_1["locations"],
            SCENARIO_1["speed"],
            SCENARIO_1["num_steps_ideal"],
            SCENARIO_1["track_length"]
        )
        actual_mean, actual_std = pset.get_intertrain_stats(location_history, SCENARIO_1["locations"])
        self.assertAlmostEqual(expected_mean, actual_mean, delta=ACCEPTABLE_EPSILON, msg=f"got get_intertrain_stats(...) mean = {actual_mean}, expected {expected_mean}")
        self.assertAlmostEqual(expected_std, actual_std, delta=ACCEPTABLE_EPSILON, msg=f"got get_intertrain_stats(...) std = {actual_std}, expected {expected_std}")

    @case_options(
        1,
        failure="get_intertrain_stats() does not return the correct mean and std",
        error="get_intertrain_stats() raised an error"
    )
    def test_get_intertrain_stats_2(self):
        expected_mean, expected_std = 5.3076923076923075, 1.7269187938956652
        location_history = pset.simulate_trains_ideal(
            SCENARIO_2["locations"],
            SCENARIO_2["speed"],
            SCENARIO_2["num_steps_ideal"],
            SCENARIO_2["track_length"]
        )
        actual_mean, actual_std = pset.get_intertrain_stats(location_history, SCENARIO_2["locations"])
        self.assertAlmostEqual(expected_mean, actual_mean, delta=ACCEPTABLE_EPSILON, msg=f"got get_intertrain_stats(...) mean = {actual_mean}, expected {expected_mean}")
        self.assertAlmostEqual(expected_std, actual_std, delta=ACCEPTABLE_EPSILON, msg=f"got get_intertrain_stats(...) std = {actual_std}, expected {expected_std}")

    @case_options(
        1,
        failure="run_monte_carlo_ideal() does not return the correct mean and std",
        error="run_monte_carlo_ideal() raised an error"
    )
    def test_run_monte_carlo_ideal_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()
        mean, std, _ = pset.run_monte_carlo_ideal(
            locations,
            speed,
            num_steps_ideal,
            track_length,
            num_trials
        )
        expected_mean, expected_std = 3.75, 0
        self.assertAlmostEqual(expected_mean, mean, delta=ACCEPTABLE_EPSILON, msg=f"got run_monte_carlo_ideal(...) mean = {mean}, expected approximately {expected_mean}")
        self.assertAlmostEqual(expected_std, std, delta=ACCEPTABLE_EPSILON, msg=f"got run_monte_carlo_ideal(...) std = {std}, expected approximately {expected_std}")

    @case_options(
        1,
        failure="run_monte_carlo_ideal() does not return the correct mean and std",
        error="run_monte_carlo_ideal() raised an error"
    )
    def test_run_monte_carlo_ideal_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()
        mean, std, _ = pset.run_monte_carlo_ideal(
            locations,
            speed,
            num_steps_ideal,
            track_length,
            num_trials
        )
        expected_mean, expected_std = 5.307692307692316, 8.881784197001252e-15
        self.assertAlmostEqual(expected_mean, mean, delta=ACCEPTABLE_EPSILON, msg=f"got run_monte_carlo_ideal(...) mean = {mean}, expected approximately {expected_mean}")
        self.assertAlmostEqual(expected_std, std, delta=ACCEPTABLE_EPSILON, msg=f"got run_monte_carlo_ideal(...) std = {std}, expected approximately {expected_std}")

@testsuite_options(4, 1)
class TestPart2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assertAcceptableRange(self, expected_mean, std, actual_mean):
        assert abs(expected_mean - actual_mean) < ACCEPTABLE_STD * std, f"Mean {actual_mean} is not within {ACCEPTABLE_STD} standard deviations of expected mean {expected_mean} (std: {std})"

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with halting does not return the correct mean",
        error="run_monte_carlo_realistic() with halting raised an error"
    )
    def test_run_monte_carlo_halting_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()

        expected_mean, expected_std = 3.552813697401451, 0.7098324417665923
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p],
            track_slowdown_fns=[pset.apply_track_halt]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with halting does not return the correct mean",
        error="run_monte_carlo_realistic() with halting raised an error"
    )
    def test_run_monte_carlo_halting_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()

        expected_mean, expected_std = 6.975007969415634, 1.2309241398642268
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p],
            track_slowdown_fns=[pset.apply_track_halt]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with gaussian slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with gaussian slowdown raised an error"
    )
    def test_run_monte_carlo_gaussian_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()

        expected_mean, expected_std = 6.34208059779687, 0.8301416893472012
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[sigma],
            track_slowdown_fns=[pset.apply_track_gaussian_slow]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with gaussian slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with gaussian slowdown raised an error"
    )
    def test_run_monte_carlo_gaussian_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()

        expected_mean, expected_std = 32.24012171627761, 7.135202604639225
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[sigma],
            track_slowdown_fns=[pset.apply_track_gaussian_slow]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with uniform slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with uniform slowdown raised an error"
    )
    def test_run_monte_carlo_uniform_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()

        expected_mean, expected_std = 7.754079150989746, 0.981115681334472
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[None],
            track_slowdown_fns=[pset.apply_track_uniform_slow]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        1,
        failure="run_monte_carlo_realistic() with uniform slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with uniform slowdown raised an error"
    )
    def test_run_monte_carlo_uniform_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()

        expected_mean, expected_std = 8.83524147434655, 1.032376846657348
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[None],
            track_slowdown_fns=[pset.apply_track_uniform_slow]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

@testsuite_options(4, 1)
class TestPart3(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assertAcceptableRange(self, expected_mean, std, actual_mean):
        assert abs(expected_mean - actual_mean) < ACCEPTABLE_STD * std, f"Mean {actual_mean} is not within {ACCEPTABLE_STD} standard deviations of expected mean {expected_mean} (std: {std})"

    @case_options(
        2,
        failure="run_monte_carlo_realistic() with no-pass and smart slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with no-pass and smart slowdown raised an error"
    )
    def test_run_monte_carlo_realistic_no_pass_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()

        expected_mean, expected_std = 9.92977008381031, 0.35156345227827984
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p, sigma],
            track_slowdown_fns=[pset.apply_track_halt, pset.apply_track_gaussian_slow],
            train_params=[None],
            train_slowdown_fns=[pset.apply_train_no_pass]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        2,
        failure="run_monte_carlo_realistic() with no-pass and smart slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with no-pass and smart slowdown raised an error"
    )
    def test_run_monte_carlo_realistic_no_pass_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()

        expected_mean, expected_std = 75.84300297480297, 8.5
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p, sigma],
            track_slowdown_fns=[pset.apply_track_halt, pset.apply_track_gaussian_slow],
            train_params=[None],
            train_slowdown_fns=[pset.apply_train_no_pass]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        2,
        failure="run_monte_carlo_realistic() with no-pass and smart slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with no-pass and smart slowdown raised an error"
    )
    def test_run_monte_carlo_realistic_no_pass_smart_1(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_1.values()

        expected_mean, expected_std = 10.26427011438588, 0.37851006828640765
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p, sigma],
            track_slowdown_fns=[pset.apply_track_halt, pset.apply_track_gaussian_slow],
            train_params=[None, 0.5],
            train_slowdown_fns=[pset.apply_train_no_pass, pset.apply_train_smart_slowdown]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)

    @case_options(
        2,
        failure="run_monte_carlo_realistic() with no-pass and smart slowdown does not return the correct mean",
        error="run_monte_carlo_realistic() with no-pass and smart slowdown raised an error"
    )
    def test_run_monte_carlo_realistic_no_pass_smart_2(self):
        locations, track_length, speed, p, sigma, num_steps_ideal, num_steps_stochastic, num_trials = SCENARIO_2.values()

        expected_mean, expected_std = 76.65753177933178, 8.665967729702702
        mean, _, _ = pset.run_monte_carlo_realistic(
            locations,
            speed,
            num_steps_stochastic,
            track_length,
            num_trials,
            track_params=[p, sigma],
            track_slowdown_fns=[pset.apply_track_halt, pset.apply_track_gaussian_slow],
            train_params=[None, 0.5],
            train_slowdown_fns=[pset.apply_train_no_pass, pset.apply_train_smart_slowdown]
        )
        self.assertAcceptableRange(expected_mean, expected_std, mean)


############################################################
# test results calculation and reporting
############################################################


class Results_600(unittest.TextTestResult):
    """Custom test result class to capture output and points."""

    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = 0
        self.max_points = 0

    def _getOptions(self, test):
        method_name = getattr(test, "_testMethodName")
        method = getattr(test, method_name)
        func = method.__func__
        points = getattr(func, "points", 0)
        failure_msg = getattr(func, "failure_message", "")
        error_msg = getattr(func, "error_message", "")
        return points, failure_msg, error_msg

    def addSuccess(self, test):
        points, _, _ = self._getOptions(test)
        self.points += points
        self.max_points += points
        return super().addSuccess(test)

    def addFailure(self, test, err):
        points, failure_msg, _ = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {failure_msg}, {err[1]}\n")
        self.max_points += points
        super().addFailure(test, err)

    def addError(self, test, err):
        points, _, error_msg = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {error_msg}, {err[1]}\n")
        self.max_points += points
        super().addError(test, err)

    def getOutput(self):
        """Return the captured output."""
        if self.points > 0:
            self.output.append(
                f"\n✅ [+{self.points}] "
                f"{'All' if self.points == self.max_points else 'Some'}"
                f" tests passed!\n"
            )
        return "\n".join(self.output)

    def getPoints(self):
        """Return the total points."""
        return self.points


if __name__ == "__main__":
    test_parts = [
        TestPart1,
        TestPart2,
        TestPart3,
    ]

    suite = unittest.TestSuite()
    for part in test_parts:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(part))
    runner = unittest.TextTestRunner(resultclass=Results_600, verbosity=2)
    result = runner.run(suite)

    output = result.getOutput()
    points_earned = round(result.getPoints(), 3)
    print(output)
    print(f"Total points: {points_earned} / {result.max_points}")
    print(f"Score: {points_earned / result.max_points:4.0%}")

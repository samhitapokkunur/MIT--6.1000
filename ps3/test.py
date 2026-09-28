# standard library
from functools import wraps
import json
import os
import unittest
from unittest.mock import MagicMock, patch

# local application
from utils import read_data_from_file
import pset


############################################################
# test case examples
############################################################


# ----- test case graphs are in tester.py ---- #


############################################################
# test case settings
############################################################


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


def normalize_list_of_tuples(list_of_tuples):
    """
    Normalize a list of tuples by sorting the inner tuples and the outer list.
    """
    normalized = []
    for inner in list_of_tuples:
        if isinstance(inner, str):
            normalized.append([inner])
        if len(inner) > 0 and isinstance(inner[0], (list, tuple)):
            normalized.append(sorted([sorted(i) for i in inner]))
        else:
            normalized.append(sorted(inner))

    return sorted(normalized)


def read_test_data(test_name, key=None):
    base_dir = os.path.join(os.path.dirname(__file__))
    with open(f"{base_dir}/tester_data/{test_name}.json", "r") as f:
        data = json.load(f)

    tester_data = data.get(key, "")
    return tester_data


############################################################
# create graph
############################################################


# add a test case for no total mapping
@testsuite_options(4, 1)
class TestPart2_2(unittest.TestCase):
    """Test graph creation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxDiff = None  # to see full diff when assertEqual fails

    def _check_create_graph(self, test_name):
        graph_filename = f"graphs/{test_name}.txt"
        expected_graph = read_test_data(test_name, key="graph")

        student_graph = pset.create_graph(graph_filename)

        expected_graph_set = {k: sorted(v) for k, v in expected_graph.items()}
        student_graph_set = {k: sorted(v) for k, v in student_graph.items()}
        self.assertDictEqual(expected_graph_set, student_graph_set)

    @case_options(
        1,
        "Your code does not create the correct graph",
        "Task test_create_graph error",
    )
    def test_create_graph_small_1(self):
        self._check_create_graph("small_1")

    @case_options(
        1,
        "Your code does not create the correct graph",
        "Task test_create_graph error",
    )
    def test_create_graph_small_2(self):
        self._check_create_graph("small_2")


############################################################
# generate all valid districts with population constraint
############################################################


@testsuite_options(4, 2)
class TestPart3_1(unittest.TestCase):
    """Test get all town combos with population constraint"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_get_town_combos(self, test_name):
        data_filename = f"data/{test_name}.json"
        graph_filename, num_districts, town_populations, _ = (
            read_data_from_file(data_filename)
        )

        expected_all_town_combo = read_test_data(
            test_name, key="all_town_combos"
        )

        graph = pset.create_graph(graph_filename)
        student_all_town_combo = pset.get_all_town_combos(
            graph, town_populations, num_districts
        )

        self.assertEqual(
            len(student_all_town_combo),
            len(expected_all_town_combo),
            "The number of generated town combos does not match the number of expected town combos.",
        )

        normalized_list1 = normalize_list_of_tuples(student_all_town_combo)
        normalized_list2 = normalize_list_of_tuples(expected_all_town_combo)

        self.assertEqual(normalized_list1, normalized_list2)

    @case_options(
        1,
        "Your code does not return the correct town combos",
        "Task test_get_town_combos_small_1 error",
    )
    def test_get_town_combos_small_1(self):
        self._check_get_town_combos("small_1")

    @case_options(
        0.5,
        "Your code does not return the correct town combos",
        "Task test_get_town_combos_small_2 error",
    )
    def test_get_town_combos_small_2(self):
        self._check_get_town_combos("small_2")

    @case_options(
        0.5,
        "Your code does not return the correct town combos",
        "Task test_get_town_combos_small_3 error",
    )
    def test_get_town_combos_small_3(self):
        self._check_get_town_combos("small_3")

    @case_options(
        0.5,
        "Your code does not return the correct town combos",
        "Task test_get_town_combos_medium_1 error",
    )
    def test_get_town_combos_medium_1(self):
        self._check_get_town_combos("medium_1")

    @case_options(
        1,
        "Your code does not return the correct town combos",
        "Task test_get_districts_mini error",
    )
    def test_get_town_combos_mini(self):
        self._check_get_town_combos("mini_1")


############################################################
# filter for connectivity criteria
############################################################


@testsuite_options(4, 2)
class TestPart3_2(unittest.TestCase):
    """Test is_compact"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        "Your code does not implement is_compact correctly",
        "Task is_compact error",
    )
    def test_is_compact_true(self):
        graph = {
            1: [2, 4, 3, 8],
            2: [1, 4],
            3: [1, 5, 6],
            4: [1, 2, 5],
            5: [4, 3, 8],
            6: [3, 7],
            7: [6, 5],
            8: [1, 5],
        }
        subgraph_nodes = (1, 2, 3, 4, 8)
        self.assertTrue(
            pset.is_compact(subgraph_nodes, graph, max_path_length=2)
        )

    @case_options(
        1,
        "Your code does not implement is_compact correctly",
        "Task is_compact error",
    )
    def test_is_compact_false(self):
        graph = {
            1: [2, 4, 3, 8],
            2: [1, 4],
            3: [1, 5, 6],
            4: [1, 2, 5],
            5: [4, 3, 8],
            6: [3, 7],
            7: [6, 5],
            8: [1, 5],
        }
        subgraph_nodes = (3, 5, 6, 7)
        self.assertFalse(
            pset.is_compact(subgraph_nodes, graph, max_path_length=2)
        )

    @case_options(
        1,
        "Your code does not implement is_compact correctly",
        "Task is_compact error",
    )
    def test_is_compact_true_multiple_paths(self):
        graph = {
            1: [2, 4, 3, 8],
            2: [1, 4],
            3: [1, 5, 6],
            4: [1, 2, 5],
            5: [4, 3, 8],
            6: [3, 7],
            7: [6, 5],
            8: [1, 5],
        }
        subgraph_nodes = (2, 4, 5, 8, 1)
        self.assertTrue(
            pset.is_compact(subgraph_nodes, graph, max_path_length=2)
        )

    @case_options(
        1,
        "Your code does not implement is_compact correctly",
        "Task is_compact error",
    )
    def test_is_compact_false_true_in_whole_graph(self):
        graph = {
            1: [2, 4, 3, 8],
            2: [1, 4],
            3: [1, 5, 6],
            4: [1, 2, 5],
            5: [4, 3, 8],
            6: [3, 7],
            7: [6, 5],
            8: [1, 5],
        }
        subgraph_nodes = (2, 4, 5, 8)
        self.assertFalse(
            pset.is_compact(subgraph_nodes, graph, max_path_length=2)
        )


############################################################
# get all valid districts
############################################################


@testsuite_options(8, 1)
class TestPart3_4(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_get_all_valid_districts(self, test_name):
        data_filename = f"data/{test_name}.json"
        graph_filename, num_districts, town_populations, _ = (
            read_data_from_file(data_filename)
        )

        expected_all_districts = read_test_data(test_name, key="all_districts")

        graph = pset.create_graph(graph_filename)
        student_all_districts = pset.get_all_valid_districts(
            graph, town_populations, num_districts, 2
        )

        self.assertEqual(
            len(student_all_districts),
            len(expected_all_districts),
            "The number of generated districts does not match the number of expected districts.",
        )

        normalized_list1 = normalize_list_of_tuples(student_all_districts)
        normalized_list2 = normalize_list_of_tuples(expected_all_districts)

        self.assertEqual(normalized_list1, normalized_list2)

    @case_options(
        1,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_mini(self):
        self._check_get_all_valid_districts("mini_1")

    @case_options(
        1,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_small_1(self):
        self._check_get_all_valid_districts("small_1")

    @case_options(
        0.5,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_small_2(self):
        self._check_get_all_valid_districts("small_2")

    @case_options(
        0.5,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_small_3(self):
        self._check_get_all_valid_districts("small_3")

    @case_options(
        0.5,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_medium_1(self):
        self._check_get_all_valid_districts("medium_1")

    @case_options(
        0.5,
        "Your code does not return the correct districts",
        "Task get_all_valid_districts error",
    )
    def test_get_all_valid_districts_medium_2(self):
        self._check_get_all_valid_districts("medium_2")


############################################################
# get all valid partitions
############################################################


@testsuite_options(8, 2)
class TestPart4_1(unittest.TestCase):
    """Testing get_all_partitions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_get_all_partitions(self, test_name):
        data_filename = f"data/{test_name}.json"
        graph_filename, num_districts, town_populations, _ = (
            read_data_from_file(data_filename)
        )

        expected_partitions = read_test_data(test_name, key="all_partitions")

        graph = pset.create_graph(graph_filename)
        valid_districts = pset.get_all_valid_districts(
            graph, town_populations, num_districts, 2
        )
        student_partitions = pset.get_all_state_partitions(
            graph, valid_districts, num_districts
        )

        self.assertEqual(
            len(student_partitions),
            len(expected_partitions),
            "The number of generated partitions does not match the number of expected partitions.",
        )

        normalized_list1 = normalize_list_of_tuples(student_partitions)
        normalized_list2 = normalize_list_of_tuples(expected_partitions)

        self.assertEqual(normalized_list1, normalized_list2)

    @case_options(
        1,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_mini(self):
        self._check_get_all_partitions("mini_1")

    @case_options(
        1,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_small_1(self):
        self._check_get_all_partitions("small_1")

    @case_options(
        0.5,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_small_2(self):
        self._check_get_all_partitions("small_2")

    @case_options(
        0.5,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_small_3(self):
        self._check_get_all_partitions("small_3")

    @case_options(
        0.5,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_medium_1(self):
        self._check_get_all_partitions("medium_1")

    @case_options(
        0.5,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_medium_2(self):
        self._check_get_all_partitions("medium_2")

    @case_options(
        1,
        "Your code does not return the correct partitions",
        "Task get_all_partitions error",
    )
    def test_get_all_partitions_no_valid(self):
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        valid_districts = [["A", "B"], ["B", "C"]]
        num_districts = 2
        student_partitions = pset.get_all_state_partitions(
            graph, valid_districts, num_districts
        )
        expected_partitions = []
        self.assertEqual(student_partitions, expected_partitions)


############################################################
# calculate vote results
############################################################


@testsuite_options(32, 2)
class TestPart4_2(unittest.TestCase):
    """Testing final outputs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_calculate_outcomes(self, test_name):
        data_filename = f"data/{test_name}.json"
        (
            graph_filename,
            num_districts,
            town_populations,
            voter_party1_proportions,
        ) = read_data_from_file(data_filename)

        expected_outcomes = read_test_data(test_name, key="outcomes_data")

        graph = pset.create_graph(graph_filename)
        student_outcomes = pset.calculate_all_partition_outcomes(
            graph, town_populations, voter_party1_proportions, num_districts, 2
        )

        self.assertCountEqual(
            expected_outcomes.keys(),
            student_outcomes.keys(),
            "Dictionaries must have the same keys for comparison.",
        )

        self.assertAlmostEqual(
            expected_outcomes["proportion_party1_total_votes"],
            student_outcomes["proportion_party1_total_votes"],
            places=3,
        )

        self.assertDictEqual(
            {
                k: v
                for k, v in expected_outcomes.items()
                if k != "proportion_party1_total_votes"
            },
            {
                k: v
                for k, v in student_outcomes.items()
                if k != "proportion_party1_total_votes"
            },
        )

    @case_options(
        1,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_mini(self):
        self._check_calculate_outcomes("mini_1")

    @case_options(
        1,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_small_1(self):
        self._check_calculate_outcomes("small_1")

    @case_options(
        0.5,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_small_2(self):
        self._check_calculate_outcomes("small_2")

    @case_options(
        0.5,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_medium_1(self):
        self._check_calculate_outcomes("medium_1")

    @case_options(
        1,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_medium_2(self):
        self._check_calculate_outcomes("medium_2")

    @case_options(
        1,
        "Your code does not return the correct outcomes",
        "Task calculate_outcomes error",
    )
    def test_calculate_outcomes_large_1(self):
        self._check_calculate_outcomes("large_1")


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
        TestPart2_2,
        TestPart3_1,
        TestPart3_2,
        TestPart3_4,
        TestPart4_1,
        TestPart4_2,
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

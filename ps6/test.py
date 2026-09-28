# standard library
from functools import wraps
import json
import os
import unittest
from unittest.mock import MagicMock, patch
import random

# local application
import pset


############################################################
# test case helpers
############################################################

def check_valid_packing(total_value, carry_on, checked_bag, items, v_cap, w_cap):
    """
    total_value, carry_on, checked_bag: output of choose_packing
    items, v_cap, w_cap: input to choose_packing
    Asserts that the packing respects the weight and volume constraints,
        the total value is correct,
        and each item appears at most once in the packing.
    """
    v_used = sum(item.get_volume() for item in carry_on)
    w_used = sum(item.get_weight() for item in checked_bag)

    # check each packed item is in the original item set
    for item in carry_on:
        assert item in items, f"Packing not valid: unknown item {item} appears in carry-on"
    for item in checked_bag:
        assert item in items, f"Packing not valid: unknown item {item} appears in checked bag"

    # check each item appears at most once
    assert len(set(carry_on).intersection(set(checked_bag))) == 0 and len(set(carry_on + checked_bag)) == len(carry_on) + len(checked_bag), \
        "Packing not valid: an item appears more than once in the packing."

    # check weight and volume constraints
    assert v_used <= v_cap, "Packing not valid: carry-on volume exceeded: %s > %s" % (v_used, v_cap)
    assert w_used <= w_cap, "Packing not valid: checked bag weight exceeded: %s > %s" % (w_used, w_cap)

    # check items in carry-on are actually carry-able
    assert all(not item.cannot_carry() for item in carry_on), \
            "Packing not valid: item %s in carry-on but cannot be carried" \
            % [item for item in carry_on if not item.cannot_carry()][0]

    # check items in check-in are actually check-able
    assert all(not item.cannot_check() for item in checked_bag), \
            "Packing not valid: item %s in check-in but cannot be checked in" \
            % [item for item in checked_bag if not item.cannot_check()][0]

    # check total value is correct
    value = sum(item.get_value() for item in carry_on) + sum(item.get_value() for item in checked_bag)
    assert value == total_value, "Total value does not match items packed: %s != %s" % (value, total_value)

def check_packing_returntype(result):
    assert isinstance(result, tuple), "choose_packing didn't return a tuple: instead returned an instance of %s." % type(result)
    assert len(result) == 3, "choose_packing didn't return 3 elements (total_value, carryon, checked). Expected %s, got %s." % (3, len(result))
    assert isinstance(result[0], int), "choose_packing's first return value (total_value) should be an int: instead got %s." % type(result[0])

    # check carryon is a list...
    assert isinstance(result[1], list), "choose_packing's second return value (carryon) should be a list of Items: instead got %s." % type(result[1])
    # ...of Items
    for item in result[1]:
        assert isinstance(item, pset.Item), f"choose_packing's second return value (carryon) should be a list of Items: element {item} is instead a {type(item)}"

    # check checked is a list...
    assert isinstance(result[2], list), "choose_packing's third return value (checked) should be a list of Items: instead got %s." % type(result[1])
    # ...of Items
    for item in result[2]:
        assert isinstance(item, pset.Item), f"choose_packing's third return value (checked) should be a list of Items: element {item} is instead a {type(item)}"

def comparable_combos(combos):
    new_combos = []
    for combo in combos:
        check_packing_returntype(combo)
        value = combo[0]
        carry_on = [f"{item}" for item in combo[1]]
        checked_bag = [f"{item}" for item in combo[2]]
        sorted_combo = (value, tuple(sorted(carry_on)), tuple(sorted(checked_bag)))
        new_combos.append(sorted_combo)
    return sorted(new_combos)

def first_missing_combo(expected_comparable, actual_comparable):
    actual_comparable = set(actual_comparable)
    for combo in expected_comparable:
        if combo not in actual_comparable:
            return (combo[0], list(combo[1]), list(combo[2]))
    return None


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


############################################################
# section 2: item class
############################################################

@testsuite_options(4, 3)
class TestPart2(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="Your code does not instantiate Items correctly",
        error="Task item_getter_methods error"
    )
    def test_item_getter_methods(self):
        item = pset.Item("jorts", 8, 2, 9, cannot_check=True)
        method_list = [
            "get_name", "get_value",
            "get_volume", "get_weight",
            "cannot_carry", "cannot_check",
            "get_info"
        ]
        expected_list = [
            "jorts", 8, 2, 9, False, True, (8, 2, 9, False, True)
        ]
        actual_list = [
            item.get_name(), item.get_value(),
            item.get_volume(), item.get_weight(),
            item.cannot_carry(), item.cannot_check(),
            item.get_info()
        ]
        for (method, expected, actual) in zip (method_list, expected_list, actual_list):
            self.assertEqual(
                expected,
                actual,
                f"Method {method} incorrect, expected: {expected}, got: {actual}",
            )

    @case_options(
        1,
        failure="Your code does not instantiate invalid Items correctly",
        error="Task item_invalid error"
    )
    def test_item_invalid(self):
        self.assertRaises(
            ValueError,
            (lambda _: pset.Item("exam solutions", 1000, 1, 1, True, True)),
            f"Incorrect behavior for invalid items",
        )

    @case_options(
        1,
        failure="Your code does not stringify Items correctly",
        error="Task item_str error"
    )
    def test_item_str(self):
        item = pset.Item("Ninja Blender", 15, 10, 7, cannot_carry=True)
        expected = "Ninja Blender: val = 15, vol = 10, weight = 7, cannot carry"
        actual = f"{item}"
        self.assertEqual(
            expected,
            actual,
            f"Incorrect item string, expected: {expected}, got: {actual}. " +
            "Make sure to NOT modify __str__"
        )
        item = pset.Item("lithium ion batteries", 2, 1, 3, cannot_check=True)
        expected = "lithium ion batteries: val = 2, vol = 1, weight = 3, cannot check"
        actual = f"{item}"
        self.assertEqual(
            expected,
            actual,
            f"Incorrect item string, expected: {expected}, got: {actual}. " +
            "Make sure to NOT modify __str__"
        )
        item = pset.Item("iPad", 30, 4, 5)
        expected = "iPad: val = 30, vol = 4, weight = 5"
        actual = f"{item}"
        self.assertEqual(
            expected,
            actual,
            f"Incorrect item string, expected: {expected}, got: {actual}. " +
            "Make sure to NOT modify __str__"
        )


############################################################
# section 3: all packing combinations
############################################################

@testsuite_options(8, 7)
class TestPart3(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_all_combinations(self, expected, actual):
        self.assertEqual(
            len(expected),
            len(actual),
            f"Incorrect number of packing combinations, expected: {len(expected)}, got: {len(actual)}"
        )
        expected_comparable = comparable_combos(expected)
        actual_comparable = comparable_combos(actual)
        self.assertEqual(
            expected_comparable,
            actual_comparable,
            f"Incorrect packing combinations, first missing combo: {first_missing_combo(expected_comparable, actual_comparable)}"
        )

    @case_options(
        3,
        failure="Your code does not generate all packing combinations correctly",
        error="Task generate_all_combinations_1 error"
    )
    def test_generate_combinations_1(self):
        shampoo = pset.Item("shampoo", 10, 5, 10)
        jorts = pset.Item("jorts", 8, 2, 8)
        items = [shampoo, jorts]
        expected = [
            (0, [], []),
            (10, [shampoo], []),
            (18, [shampoo, jorts], []),
            (18, [shampoo], [jorts]),
            (10, [], [shampoo]),
            (18, [jorts], [shampoo]),
            (18, [], [shampoo, jorts]),
            (8, [jorts], []),
            (8, [], [jorts]),
        ]
        actual = pset.all_packing_combinations(items)
        self.check_all_combinations(expected, actual)

    @case_options(
        2,
        failure="Your code does not generate the right number of packing combinations",
        error="Task generate_all_combinations_2 error"
    )
    def test_generate_combinations_2(self):
        items = [
            pset.Item("shampoo", 10, 5, 10, cannot_carry=True),
            pset.Item("pomade", 8, 3, 4),
            pset.Item("brush", 6, 4, 4, cannot_check=True),
            pset.Item("wave cap", 4, 1, 1),
        ]
        actual = pset.all_packing_combinations(items)
        self.assertEqual(
            3**4,
            len(actual),
            f"Incorrect number of packing combinations, expected: {3**4}, got: {len(actual)}"
        )

    @case_options(
        1,
        failure="Your code does not generate all packing combinations correctly (for 1 item)",
        error="Task generate_all_combinations_3 error"
    )
    def test_generate_combinations_3(self):
        shampoo = pset.Item("shampoo", 10, 5, 10),
        items = [shampoo]
        expected = [
            (0, [], []),
            (10, [shampoo], []),
            (10, [], [shampoo]),
        ]
        actual = pset.all_packing_combinations(items)
        self.check_all_combinations(expected, actual)

    @case_options(
        1,
        failure="Your code does not generate all packing combinations correctly (for 0 items)",
        error="Task generate_all_combinations_4 error"
    )
    def test_generate_combinations_3(self):
        items = []
        expected = [(0, [], [])]
        actual = pset.all_packing_combinations(items)
        self.assertEqual(
            expected,
            actual,
            f"Incorrect packing combinations, expected: {expected}, got: {actual}"
        )


############################################################
# section 4: brute-force solution
############################################################

@testsuite_options(8, 6)
class TestPart4(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_choose_packing_all_fit error"
    )
    def test_choose_packing_all_fit(self):
        # test case where all items fit, no constraints
        items = [
            pset.Item('shampoo', 10, 5, 10),
            pset.Item('Introduction to Computation and Programming Using Python, Third Edition, With Application to Computational Modeling and Understanding Data by John Guttag', 15, 10, 20),
            pset.Item('pokemon plushies', 7, 3, 5),
            pset.Item('jorts', 8, 2, 8),
        ]

        v_cap = 15
        w_cap = 25

        result = pset.choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 40) # check for optimality

    @case_options(
        1,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_choose_packing_first_not_optimal error"
    )
    def test_choose_packing_first_not_optimal(self):
        # test case where it's not optimal to take the first item
        items = [
            pset.Item('spray tan spray', 8, 6, 6, cannot_carry=True),
            pset.Item('bucket hat collection', 10, 5, 5),
            pset.Item('baby shark singing plushie', 9, 5, 5),
        ]

        v_cap = 0
        w_cap = 10

        result = pset.choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 19)

    @case_options(
        1,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_choose_packing_carryon_greedy_not_optimal error"
    )
    def test_choose_packing_carryon_greedy_not_optimal(self):
        # test case where greedy solution that puts items in carry-on till full isn't optimal
        items = [
            pset.Item('7.012 textbook', 1, 3, 10),
            pset.Item('Pocari sweat', 1, 3, 3, cannot_carry=True),
            pset.Item('extra underwear', 10, 6, 20), # greedy would put 1st two in carry-on, but optimal is to put this one in carry-on
            pset.Item('laptop', 5, 5, 10),
            pset.Item('extra shoes', 5, 5, 4),
            pset.Item('6-7 packs of gum', 2, 5, 4),
        ]

        v_cap = 6  # Carry-on volume capacity
        w_cap = 15  # Checked bag weight capacity

        result = pset.choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 20)

    @case_options(
        1,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_choose_packing_checked_greedy_not_optimal error"
    )
    def test_choose_packing_checked_greedy_not_optimal(self):
        # test case where greedily filling the checked first is not optimal
        items = [
            pset.Item('Curious george plushie', 1, 3, 5),
            pset.Item("scented hand lotion", 1, 3, 5),
            pset.Item('camera', 10, 6, 10), # greedy would put 1st two in checked, but optimal is to put this one in checked
            pset.Item('portable battery', 5, 5, 10),
        ]

        v_cap = 2
        w_cap = 10

        result = pset.choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 10)

    @case_options(
        2,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_choose_packing_constrained error"
    )
    def test_choose_packing_constrained(self):
        # test case with significant constraints
        items = [
            pset.Item('Curious george plushie', 1, 3, 5),
            pset.Item("scented hand lotion", 1, 3, 5),
            pset.Item('camera', 10, 6, 10, cannot_check=True), # greedy would put 1st two in checked, but optimal is to put this one in checked
            pset.Item('portable battery', 5, 5, 10, cannot_carry=True),
            pset.Item('extra shoes', 5, 5, 4, cannot_carry=True),
        ]

        v_cap = 5
        w_cap = 10

        result = pset.choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 7)


############################################################
# section 5: dynamic programming solution
############################################################

@testsuite_options(8, 3)
class TestPart5(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        3,
        failure="Your code does not generate an optimal packing solution",
        error="Task test_dp_choose_packing error"
    )
    def test_dp_choose_packing(self):
        # large test case where recursive solution would hang for a long time
        random.seed(21)
        # 20 random items
        items = [pset.Item(f'Item {i}', random.randint(5, 20), random.randint(2, 10), random.randint(2, 10)) for i in range(20)]
        v_cap = 50
        w_cap = 50

        result = pset.dp_choose_packing(items, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, v_cap, w_cap)
        self.assertEqual(result[0], 267)


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
        TestPart2,
        TestPart3,
        TestPart4,
        TestPart5
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

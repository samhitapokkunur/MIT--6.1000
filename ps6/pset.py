"""
6.1000 Fall 2025
Problem Set 6

Please fill out the following info:
Name:
Kerberos:
Approximate time spent (HH:MM):
"""

import random
import time


############################################################
# item class
############################################################

class Item(object):
    """
    Represents an item that can be packed into luggage.

    Each item has:
    - name: a string identifier
    - val: an integer value
    - vol: an integer volume (in liters)
    - weight: an integer weight (in pounds)
    - cannot_carry: a boolean indicating if the item cannot go in carry-on
    - cannot_check: a boolean indicating if the item cannot go in checked bag

    Note: cannot_carry and cannot_check cannot both be True.
    """

    def __init__(self, name, val, vol, weight, cannot_carry=False,
                 cannot_check=False):
        """
        Initialize an Item.

        Parameters:
            name (str): the name of the item
            val (int): the value of the item
            vol (int): the volume of the item in liters
            weight (int): the weight of the item in pounds
            cannot_carry (bool): True if item cannot go in carry-on
            cannot_check (bool): True if item cannot go in checked bag

        Raises:
            ValueError: if both cannot_carry and cannot_check are True
        """
        # Check if both cannot_carry and cannot_check are True
        if cannot_carry and cannot_check:
            raise ValueError("Item cannot have both cannot_carry and cannot_check set to True")

        # Initialize all private fields
        self._name = name
        self._val = val
        self._vol = vol
        self._weight = weight
        self._cannot_carry = cannot_carry
        self._cannot_check = cannot_check

    def get_name(self):
        """
        Returns the name of the item.

        Returns:
            str: the item's name
        """
        return self._name

    def get_value(self):
        """
        Returns the value of the item.

        Returns:
            int: the item's value
        """
        return self._val

    def get_volume(self):
        """
        Returns the volume of the item.

        Returns:
            int: the item's volume in liters
        """
        return self._vol

    def get_weight(self):
        """
        Returns the weight of the item.

        Returns:
            int: the item's weight in pounds
        """
        return self._weight

    def cannot_carry(self):
        """
        Returns whether the item cannot be placed in carry-on.

        Returns:
            bool: True if item cannot go in carry-on, False otherwise
        """
        return self._cannot_carry

    def cannot_check(self):
        """
        Returns whether the item cannot be placed in checked bag.

        Returns:
            bool: True if item cannot go in checked bag, False otherwise
        """
        return self._cannot_check

    def get_info(self):
        """
        Returns a tuple containing all item information.

        Returns:
            tuple: (value, volume, weight, cannot_carry, cannot_check)
        """
        return (self._val, self._vol, self._weight, self._cannot_carry, self._cannot_check)

    def __str__(self):
        """
        Returns a string representation of the item.
        DO NOT MODIFY THIS METHOD.
        """
        if self._cannot_carry:
            constraint = ', cannot carry'
        elif self._cannot_check:
            constraint = ', cannot check'
        else:
            constraint = ''

        return (f'{self._name}: val = {self._val}, vol = {self._vol}, ' +
                f'weight = {self._weight}{constraint}')

    def __repr__(self):
        """
        Returns a string representation of the item (for use in lists).
        DO NOT MODIFY THIS METHOD.
        """
        return self.__str__()


############################################################
# all packing combinations helper
############################################################

def all_packing_combinations(items):
    """
    Generates all possible ways to pack the given items, ignoring capacity
    constraints and item restrictions. This is a helper function for the
    brute-force solution.

    For each item, considers three possibilities: don't pack it, put it in carry-on,
    or put it in checked bag. Recursively generates all combinations.

    This function should NOT check capacity constraints or item restrictions.
    Those are validated later.

    Parameters:
        items (list): a list of Item objects to consider packing

    Returns:
        a list of tuples, where each tuple has the form (total_value, carry_on, checked_bag)
        and represents a single combination where:
              - total_value (int): sum of values of all packed items
              - carry_on (list): list of Item objects in carry-on
              - checked_bag (list): list of Item objects in checked bag
    """
    # Base case: no items left to consider
    if len(items) == 0:
        return [(0, [], [])]

    # Recursive case: consider the first item
    first_item = items[0]
    remaining_items = items[1:]

    # Get all combinations for the remaining items
    sub_combinations = all_packing_combinations(remaining_items)

    all_combinations = []

    # For each combination of remaining items, generate three new combinations
    for total_value, carry_on, checked_bag in sub_combinations:
        # Option 1: Don't pack the first item
        all_combinations.append((total_value, carry_on[:], checked_bag[:]))

        # Option 2: Put the first item in carry-on
        new_carry_on = carry_on[:] + [first_item]
        new_value = total_value + first_item.get_value()
        all_combinations.append((new_value, new_carry_on, checked_bag[:]))

        # Option 3: Put the first item in checked bag
        new_checked_bag = checked_bag[:] + [first_item]
        new_value = total_value + first_item.get_value()
        all_combinations.append((new_value, carry_on[:], new_checked_bag))

    return all_combinations


############################################################
# brute-force solution
############################################################

def choose_packing(items, v_cap, w_cap):
    """
    Determines the packing with the maximum total value that can be achieved
    given the volume and weight capacities and the item restrictions.

    Must be implemented using brute-force and with the
    `all_packing_combinations` helper function.

    Parameters:
        items (list): a list of Item objects
        v_cap (int): volume capacity of all items in carry on bag
        w_cap (int): weight capacity of all items in checked bag

    Returns:
        tuple: (total_value, carry_on, checked_bag) where:
            total_value (int): the max achievable value of the packed Items
            carry_on (list): list of Item objects that achieve the optimal value
            checked_bag (list): list of Item objects that achieve the optimal value
    """
    # Generate all possible packing combinations
    all_combinations = all_packing_combinations(items)

    # Track the best valid combination
    best_value = 0
    best_carry_on = []
    best_checked_bag = []

    # Filter for valid combinations and find the maximum value
    for total_value, carry_on, checked_bag in all_combinations:
        # Check if this combination is valid
        if is_valid_packing(carry_on, checked_bag, v_cap, w_cap):
            # If this combination has higher value than current best, update
            if total_value > best_value:
                best_value = total_value
                best_carry_on = carry_on
                best_checked_bag = checked_bag

    return (best_value, best_carry_on, best_checked_bag)


def is_valid_packing(carry_on, checked_bag, v_cap, w_cap):
    """
    Helper function to check if a packing combination is valid.

    A packing is valid if:
    1. Total volume of carry-on items <= v_cap
    2. Total weight of checked bag items <= w_cap
    3. No item in carry-on has cannot_carry=True
    4. No item in checked bag has cannot_check=True

    Parameters:
        carry_on (list): list of Item objects in carry-on
        checked_bag (list): list of Item objects in checked bag
        v_cap (int): volume capacity of carry-on bag
        w_cap (int): weight capacity of checked bag

    Returns:
        bool: True if the packing is valid, False otherwise
    """
    # Check carry-on volume capacity
    total_carry_volume = sum(item.get_volume() for item in carry_on)
    if total_carry_volume > v_cap:
        return False

    # Check carry-on item restrictions
    for item in carry_on:
        if item.cannot_carry():
            return False

    # Check checked bag weight capacity
    total_checked_weight = sum(item.get_weight() for item in checked_bag)
    if total_checked_weight > w_cap:
        return False

    # Check checked bag item restrictions
    for item in checked_bag:
        if item.cannot_check():
            return False

    return True

############################################################
# dynamic programming solution
############################################################

def dp_choose_packing(items, v_cap, w_cap):
    """
    Determines the packing with the maximum total value that can be achieved
    given the volume and weight capacities and the item restrictions.

    Must be implemented using dynamic programming.

    Parameters:
        items (list): a list of Item objects
        v_cap (int): volume capacity of all items in carry on bag
        w_cap (int): weight capacity of all items in checked bag

    Returns:
        tuple: (total_value, carry_on, checked_bag) where:
            total_value (int): the max achievable value of the packed Items
            carry_on (list): list of Item objects that achieve the optimal value
            checked_bag (list): list of Item objects that achieve the optimal value
    """
    # Memoization dictionary: key is (item_index, remaining_v_cap, remaining_w_cap)
    memo = {}

    def dp_helper(index, v_remaining, w_remaining):
        """
        Recursive helper function that returns the best packing for items[index:]
        given remaining capacities.

        Parameters:
            index (int): current item index to consider
            v_remaining (int): remaining volume capacity in carry-on
            w_remaining (int): remaining weight capacity in checked bag

        Returns:
            tuple: (total_value, carry_on, checked_bag)
        """
        # Base case: no more items to consider
        if index >= len(items):
            return (0, [], [])

        # Check if we've already solved this subproblem
        if (index, v_remaining, w_remaining) in memo:
            return memo[(index, v_remaining, w_remaining)]

        current_item = items[index]

        # Initialize with the "don't pack" option
        best_value, best_carry, best_checked = dp_helper(index + 1, v_remaining, w_remaining)

        # Option 1: Try putting current item in carry-on
        if (not current_item.cannot_carry() and
            current_item.get_volume() <= v_remaining):
            # Recursively solve for remaining items with reduced carry-on capacity
            sub_value, sub_carry, sub_checked = dp_helper(
                index + 1,
                v_remaining - current_item.get_volume(),
                w_remaining
            )
            total_value = current_item.get_value() + sub_value

            # Update best solution if this is better
            if total_value > best_value:
                best_value = total_value
                best_carry = [current_item] + sub_carry
                best_checked = sub_checked

        # Option 2: Try putting current item in checked bag
        if (not current_item.cannot_check() and
            current_item.get_weight() <= w_remaining):
            # Recursively solve for remaining items with reduced checked bag capacity
            sub_value, sub_carry, sub_checked = dp_helper(
                index + 1,
                v_remaining,
                w_remaining - current_item.get_weight()
            )
            total_value = current_item.get_value() + sub_value

            # Update best solution if this is better
            if total_value > best_value:
                best_value = total_value
                best_carry = sub_carry
                best_checked = [current_item] + sub_checked

        # Store result in memo before returning
        memo[(index, v_remaining, w_remaining)] = (best_value, best_carry, best_checked)

        return (best_value, best_carry, best_checked)

    # Start the recursion from index 0 with full capacities
    return dp_helper(0, v_cap, w_cap)

############################################################
# example tests
############################################################

def example_test(size = "small", dp = False):
    random.seed(21)

    num_items_dict = {"small": 5, "medium": 12, "large": 20}
    cap_size_dict = {"small": 12, "medium": 30, "large": 50}

    items = []
    for i in range(num_items_dict[size]):
        item = Item(f"Item {i}", random.randint(5, 20),
            random.randint(2, 10), random.randint(2, 10))
        items.append(item)
    v_cap = cap_size_dict[size]
    w_cap = cap_size_dict[size]

    if dp:
        packing_function = dp_choose_packing
    else:
        packing_function = choose_packing

    start_time = time.time()
    result = packing_function(items, v_cap, w_cap)
    end_time = time.time()
    run_time = end_time - start_time
    return result, run_time


def format_results(result, run_time, size, dp):
    output = []
    output.append(f"TEST: {size}, {'dp' if dp else 'brute force'}")
    output.append(f"run time: {run_time:5f}s")
    output.append(f"total_value: {result[0]}")
    output.append(f"carry on: {result[1]}")
    output.append(f"checked bag: {result[2]}")
    output.append("")
    return "\n".join(output)


############################################################
# experimental analysis
############################################################

import matplotlib.pyplot as plt

def experiment1_runtime_vs_items():
    """
    Experiment 1: Compare runtime of brute-force vs DP as a function of number of items.
    """
    num_items_list = list(range(5, 15))
    v_cap = 30
    w_cap = 30
    num_trials = 3

    brute_force_times = []
    dp_times = []

    for num_items in num_items_list:
        brute_force_avg = 0
        dp_avg = 0

        for trial in range(num_trials):
            random.seed(21 + trial)
            items = []
            for i in range(num_items):
                item = Item(f"Item {i}", random.randint(5, 20),
                    random.randint(2, 10), random.randint(2, 10))
                items.append(item)

            start = time.time()
            choose_packing(items, v_cap, w_cap)
            brute_force_avg += (time.time() - start)

            start = time.time()
            dp_choose_packing(items, v_cap, w_cap)
            dp_avg += (time.time() - start)

        brute_force_times.append(brute_force_avg / num_trials)
        dp_times.append(dp_avg / num_trials)

    plt.figure(figsize=(10, 6))
    plt.plot(num_items_list, brute_force_times, 'o-', label='Brute-force', linewidth=2, markersize=8)
    plt.plot(num_items_list, dp_times, 's-', label='Dynamic Programming', linewidth=2, markersize=8)
    plt.xlabel('Number of Items', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime Comparison: Brute-force vs Dynamic Programming', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Experiment 1 complete: Runtime vs Number of Items")


def experiment2_dp_vs_discreteness():
    """
    Experiment 2: Compare DP runtime as a function of value discreteness.
    Discreteness is varied by constraining volumes and weights to multiples of different numbers.
    """
    num_items = 15
    v_cap = 1024
    w_cap = 1024
    num_trials = 3


    divisors = [2**i for i in range(8)]
    dp_times = []
    discreteness_labels = []

    for divisor in divisors:
        dp_avg = 0

        for trial in range(num_trials):
            random.seed(42 + trial)
            items = []
            for i in range(num_items):
                item = Item(f"Item {i}",
                           random.randint(1, 100),
                           divisor * random.randint(1, v_cap // divisor // 4),
                           divisor * random.randint(1, w_cap // divisor // 4))
                items.append(item)

            start = time.time()
            dp_choose_packing(items, v_cap, w_cap)
            dp_avg += (time.time() - start)

        dp_times.append(dp_avg / num_trials)
        if divisor == 1:
            discreteness_labels.append("Any int")
        else:
            discreteness_labels.append(f"Multiple of {divisor}")

    plt.figure(figsize=(10, 6))
    plt.plot(divisors, dp_times, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Discreteness (Divisor)', fontsize=12)
    plt.ylabel('DP Runtime (seconds)', fontsize=12)
    plt.title('DP Runtime vs Volume/Weight Discreteness (12 items)', fontsize=14)
    plt.xscale('log')
    plt.xticks(divisors, discreteness_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Experiment 2 complete: DP Runtime vs Volume/Weight Discreteness")


if __name__ == "__main__":
    # Uncomment to run experiments

    # experiment1_runtime_vs_items()
    # experiment2_dp_vs_discreteness()

    # # small brute force
    # result, run_time = example_test(size="small", dp=False)
    # print(format_results(result, run_time, size="small", dp=False))

    # # small dp
    # result, run_time = example_test(size="small", dp=True)
    # print(format_results(result, run_time, size="small", dp=True))

    # # medium brute force
    # result, run_time = example_test(size="medium", dp=False)
    # print(format_results(result, run_time, size="medium", dp=False))

    # # medium dp
    # result, run_time = example_test(size="medium", dp=True)
    # print(format_results(result, run_time, size="medium", dp=True))

    # # large brute force (likely will not run to completion)
    # result, run_time = example_test(size="large", dp=False)
    # print(format_results(result, run_time, size="large", dp=False))

    # # large dp
    # result, run_time = example_test(size="large", dp=True)
    # print(format_results(result, run_time, size="large", dp=True))

    pass

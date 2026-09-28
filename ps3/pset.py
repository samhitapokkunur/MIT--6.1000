"""
6.1000 Fall 2025
Problem Set 3

Please fill out the following info:
Name: Samhita Pokkunuri
Kerberos: samhitap
Approximate time spent (HH:MM): 10:00
"""

from utils import (
    plot_voronoi_from_graph,
    plot_graph_and_voronoi,
    read_data_from_file,
)
import os
import random
import sys

sys.setrecursionlimit(10000)


############################################################
# create graph
############################################################


def create_graph(file_name):
    """
    Create a graph representation from a file.

    Parameters:
        file_name (str): the name of the text file to be read and interpreted.

    Return a graph mapping town names to an adjacency list.
    """
    graph = {}

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split the line into two towns separated by a comma
            a, b = [x.strip() for x in line.split(",")]

            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

    # sort the graph by town names
    sorted_items = sorted(graph.items())

    # sort the neighbor lists for each town
    for town, neighbors in sorted_items:
        sorted_neighbors = sorted(list(neighbors))
        graph[town] = sorted_neighbors

    return graph

    raise NotImplementedError



############################################################
# generate all valid districts with population constraint
############################################################


def town_combos_dfs_helper(
    all_town_combos,
    town_info,
    remaining_towns,
    current_combination,
    current_population,
):
    """
    DFS Helper function for get_all_town_combos

    Parameters:
        all_town_combos (list): A list of all valid combos,
            where a combination is represented by a list of town names.
        town_info (dict): A dictionary with the following keys:
            "town_names": maps to a list,
            "town_populations": maps to a dict,
            "population_lower": maps to a float of the lower bound,
            "population_upper": maps to a float of the upper bound
        remaining_towns (list): All towns not considered yet.
        current_combination (list): All towns in current combination.
        current_population (int): Sum of populations for towns in current_combination.
    """
    if current_population > town_info["population_upper"]:
        return

    # check if all towns have been considered
    if not remaining_towns:
        # verify population is within valid range
        if (
            town_info["population_lower"]
            <= current_population
            <= town_info["population_upper"]
        ):
            # add valid combination
            all_town_combos.append(tuple(current_combination))
        return

    # choose the next town to process
    next_town = remaining_towns[0]
    remaining_towns_after = remaining_towns[1:]
    next_town_pop = town_info["town_populations"][next_town]

    # dfs branch 1: include the next town
    current_combination.append(next_town)
    new_population = current_population + next_town_pop

    # recurse with the town included
    town_combos_dfs_helper(
        all_town_combos,
        town_info,
        remaining_towns_after,
        current_combination,
        new_population,
    )

    # backtrack to explore other options
    current_combination.pop()

    # dfs branch 2: exclude the next town
    town_combos_dfs_helper(
        all_town_combos,
        town_info,
        remaining_towns_after,
        current_combination,
        current_population,
    )

    #raise notimplementederror


def get_all_town_combos(graph, town_populations, num_districts):
    """
    Find all possible town combos with a total population within 10% of the average across districts.

    Parameters:
        graph (dict(str, list[str])): An adjacency dictionary representation of the graph.
        town_populations (dict(str, int)): A dictionary mapping a town to its population.
        num_districts (int): The number of districts in the state.

    Return a list of town combos, each represented as a tuple of town names.
    """
    total_population = 0
    for town_pop in town_populations.values():
        total_population += town_pop
    avg_dist_pop = total_population / num_districts

    town_info = {
        "town_populations": town_populations,
        "population_lower": avg_dist_pop * 0.9,
        "population_upper": avg_dist_pop * 1.1,
    }

    all_town_combos = []
    town_combos_dfs_helper(
        all_town_combos,
        town_info,
        list(graph.keys()),
        [],
        0,
    )
    return all_town_combos


############################################################
# filter for connectivity criteria
############################################################


def is_compact(subgraph_nodes, graph, max_path_length):
    """
    Check if a given set of nodes is compact.

    Parameters:
        subgraph_nodes (tuple[str]): The subgraph to check.
        graph (dict[str, list[str]]): A dictionary mapping the town names to an adjacency.
        max_path_length (int): The maximum number of edges between any two nodes in the subgraph.

    Return True if the subgraph is compact, False otherwise.
    """
    if len(subgraph_nodes) <= 1:
        return True

    # loop through each town in the district as a starting point
    for start_town in subgraph_nodes:
        # setup bfs queue (town, path_length)
        queue = [(start_town, 0)]
        # track visited towns
        visited = {start_town}
        # count how many towns are reachable
        num_reached_targets = 0

        # bfs loop
        while queue:
            # get next town
            current_town, current_dist = queue.pop(0)
            # explore neighbors
            for neighbor in graph.get(current_town, []):
                # skip towns not in district or already visited
                if neighbor in subgraph_nodes and neighbor not in visited:
                    new_dist = current_dist + 1
                    # stop if path exceeds allowed length
                    if new_dist > max_path_length:
                        return False
                    visited.add(neighbor)
                    queue.append((neighbor, new_dist))
                    num_reached_targets += 1
                    # stop early if all towns are reached
                    if num_reached_targets == len(subgraph_nodes) - 1:
                        break

            # stop bfs early if district fully connected from start_town
            if num_reached_targets == len(subgraph_nodes) - 1:
                break

        # if any town couldn’t be reached, district isn’t contiguous
        if num_reached_targets < len(subgraph_nodes) - 1:
            return False

    # all towns are within max_path_length of each other
    return True

    raise NotImplementedError


def get_all_valid_districts(
    graph, town_population, num_districts, max_path_length
):
    """
    Find all districts that satisfy the three constraints.

    Parameters:
        graph (dict[str, list[str]]): A dictionary mapping town names to an adjacency list.
        town_population (dict[str, int]): A dictionary mapping a town to its population.
        num_districts (int): The number of districts in the state.
        max_path_length (int): The maximum path length allowed for partitions.

    Return a list of valid districts, each represented as a tuple of town names.
    """
    all_population_combos = get_all_town_combos(
    graph, town_population, num_districts
    )

    valid_districts = []

    # filter combinations by compactness
    # is_compact checks both contiguity and max path length
    for combo in all_population_combos:
        if is_compact(combo, graph, max_path_length):
            valid_districts.append(combo)

    # return all valid districts
    return valid_districts

    #raise NotImplementedError



############################################################
# get all valid partitions
############################################################


def state_partitions_dfs_helper(
    all_partitions,
    district_info,
    remaining_districts,
    current_partition,
    used_towns,
):
    """
    DFS Helper function for get_all_state_partitions

    Parameters:
        all_partitions (list): A list of all valid partitions,
            where a partition is represented as a list of districts.
        town_info (dict): A dictionary with the following keys:
            "all_towns": maps to a list of all town names,
            "num_districts": maps to the number of districts
        remaining_districts (list): All districts not considered yet.
        current_partition (list): All towns in current combination.
        used_towns (list): All towns already used in the current_partition.
    """
    all_towns = district_info["all_towns"]
    num_districts = district_info["num_districts"]

    # stop if we already have too many districts
    if len(current_partition) > num_districts:
        return

    # base case: valid partition found
    # check 1: correct number of districts
    # check 2: all towns are covered
    if (len(current_partition) == num_districts and
        len(used_towns) == len(all_towns)):

    # add valid partition to results
        all_partitions.append(list(current_partition))
        return

    # base case: no more districts left to consider
    if not remaining_districts:
        return

    # dfs step: pick the next district
    district = remaining_districts[0]
    remaining_districts_after = remaining_districts[1:]

    # check for overlap between used towns and new district
    is_overlap = False # ensuring not reconsidering same points
    newly_used_towns = []

    for town in district:
        if town in used_towns:
            is_overlap = True #removing if already in used towns
            break
        newly_used_towns.append(town)

    # decision 1: include the district (only if no overlap)
    if not is_overlap:
        new_partition = current_partition + [district]
        new_used_towns = used_towns + newly_used_towns

        state_partitions_dfs_helper(
            all_partitions,
            district_info,
            remaining_districts_after,
            new_partition,
            new_used_towns,
        )

    # decision 2: skip the district aka if there is overlap
    # skip if not enough districts left to reach target count
    if len(current_partition) + len(remaining_districts) - 1 >= num_districts:
        state_partitions_dfs_helper(
            all_partitions,
            district_info,
            remaining_districts_after,
            current_partition,  # unchanged
            used_towns,         # unchanged
        )

#raise notimplementederror


def get_all_state_partitions(graph, all_valid_districts, num_districts):
    """
    Find all possible partitions of districts.

    Parameters:
        graph (dict): An adjacency dictionary representation of the graph.
        all_valid_districts (list): A list of valid districts.
        num_districts (int): The number of districts in the state.

    Return a list of partitions, each represented as a list of districts.
    """
    all_valid_districts = sorted(all_valid_districts, key=len, reverse=True)
    district_info = {
        "all_towns": list(graph.keys()),
        "num_districts": num_districts,
    }

    all_partitions = []
    state_partitions_dfs_helper(
        all_partitions,
        district_info,
        all_valid_districts,
        [],
        [],
    )

    return all_partitions


############################################################
# calculate vote results
############################################################


def calculate_all_partition_outcomes(
    graph, town_populations, voter_proportions, num_districts, max_path_length
):
    """
    Calculate the desired voting outcomes, as described in the pset.

    Parameters:
        graph (dict): The graph representing the towns and their connections.
        town_populations (dict): A dictionary mapping a town to its population.
        voter_proportions (dict): A dictionary mapping a town to its voter proportion.
        num_districts (int): The number of districts to partition the towns into.
        max_path_length (int): The maximum path length allowed for partitions.

    Return a dictionary with the statistics described in the pset.
    """
    all_valid_districts = get_all_valid_districts(
        graph, town_populations, num_districts, max_path_length
    )

    # find all valid partitions that cover every town once
    # assumes get_all_state_partitions is defined
    all_partitions = get_all_state_partitions(
        graph, all_valid_districts, num_districts
    )

    # calculate total state vote proportions
    total_state_votes = 0
    total_party1_votes = 0

    # use population for votes
    for town, population in town_populations.items():
        votes_in_town = population
        party1_proportion = voter_proportions[town]

        total_state_votes += votes_in_town
        total_party1_votes += votes_in_town * party1_proportion

    proportion_party1_total_votes = 0.0
    if total_state_votes > 0:
        proportion_party1_total_votes = total_party1_votes / total_state_votes

    # evaluate each partition and count outcomes
    partition_party1_wins = 0
    partition_party2_wins = 0
    partition_ties = 0

    for partition in all_partitions:
        # each partition is a list of districts
        party1_seats = 0
        party2_seats = 0

        # determine winner for each district
        for district in partition:
            district_votes = 0
            district_party1_votes = 0

            for town in district:
                votes_in_town = town_populations[town]
                party1_proportion = voter_proportions[town]

                district_votes += votes_in_town
                district_party1_votes += votes_in_town * party1_proportion

            # decide district winner (> 50% wins)
            if district_votes > 0:
                if district_party1_votes > district_votes * 0.5:
                    party1_seats += 1
                elif district_party1_votes < district_votes * 0.5:
                    party2_seats += 1
                # ties are ignored

        # decide overall partition winner
        if party1_seats > party2_seats:
            partition_party1_wins += 1
        elif party2_seats > party1_seats:
            partition_party2_wins += 1
        else:
            partition_ties += 1

    # assemble final results
    results = {
        "proportion_party1_total_votes": round(proportion_party1_total_votes, 3),
        "num_possible_districts": len(all_valid_districts),
        "num_partitions": len(all_partitions),
        "partition_party1_wins": partition_party1_wins,
        "partition_party2_wins": partition_party2_wins,
        "partition_ties": partition_ties
    }

    return results

    raise NotImplementedError



############################################################
# functions to manual test
############################################################
def manual_get_all_town_combos(file_name):
    """
    Prints all town combos for a small testcase.

    Parameters:
        file_name (str): Name of the file without the 'json'.
    """
    file_path_json = f"data/{file_name}.json"
    graph_filepath, num_districts, town_populations, _ = read_data_from_file(
        file_path_json
    )

    graph = create_graph(graph_filepath)

    all_town_combos = get_all_town_combos(
        graph, town_populations, num_districts
    )

    print(f"Number of town combos: {len(all_town_combos)}")
    print(f"All town combos: {all_town_combos}")


def manual_test_print_files(file_name, print_json=True, print_txt=True):
    """
    Given a file_name, optionally prints content of .json and .txt file.

    Parameters:
        file_name (str): Name of the file without 'txt' or 'json'.
        print_txt (boolean): Indicates whether the graph edges should be printed.
        print_json (boolean): Indicates whether the town information should be printed.
    """
    file_path_json = f"data/{file_name}.json"
    data = read_data_from_file(file_path_json)
    file_path_txt = f"{os.path.dirname(__file__)}/{data[0]}"

    if print_json:
        print(f"{file_name}.json:\n{data}\n\n")

    with open(file_path_txt, "r") as file:
        content = file.read()
        if print_txt:
            print(f"{file_name}.txt:\n{content}\n\n")


def manual_get_all_valid_districts(file_name, max_path_length=2):
    """
    Prints all valid district for a small testcase and plots the voronoi diagram.

    Parameters:
        file_name (str): Name of the file without 'txt' or 'json'.
        max_path_length (int): The 'compactness' constraint, aka max distance between two
    """
    file_path_json = f"data/{file_name}.json"
    graph_filepath, num_districts, town_populations, _ = read_data_from_file(
        file_path_json
    )

    graph = create_graph(graph_filepath)

    all_districts = get_all_valid_districts(
        graph, town_populations, num_districts, max_path_length
    )

    plot_voronoi_from_graph(graph, [], file_path_json, show_plot=True)
    print(f"Number of valid districts: {len(all_districts)}")
    print(f"All valid districts: {all_districts}")


def manual_get_all_valid_partitions(file_name, max_path_length=2, plot_k=5):
    """
    Prints all valid partitions for a small testcase and plots the color-coded voronoi diagram.

    Parameters:
        file_name (str): Name of the file without 'txt' or 'json'.
        max_path_length (int): The 'compactness' constraint, aka max distance between two
        plot_k (int): The number of partitions to plot.
    """
    file_path_json = f"data/{file_name}.json"
    graph_filepath, num_districts, town_populations, _ = read_data_from_file(
        file_path_json
    )

    graph = create_graph(graph_filepath)

    all_districts = get_all_valid_districts(
        graph, town_populations, num_districts, max_path_length
    )
    all_partitions = get_all_state_partitions(
        graph, all_districts, num_districts
    )
    print(f"Number of valid districts: {len(all_districts)}")
    print(f"Number of valid partitions: {len(all_partitions)}")

    random.seed(42)
    random.shuffle(all_partitions)

    for partition in all_partitions[:plot_k]:
        print("Partition:", partition)
        plot_graph_and_voronoi(graph, partition, file_path_json, show_plot=True)


def manual_calculate_voting_outcomes(file_name, max_path_length=2):
    """
    Prints the voting outcomes for all partitions for a small testcase.

    Parameters:
        file_name (str): Name of the file without 'txt' or 'json'.
        max_path_length (int): The 'compactness' constraint, aka max distance between two
    """
    file_path_json = f"data/{file_name}.json"
    (
        graph_filepath,
        num_districts,
        town_populations,
        voter_party1_proportions,
    ) = read_data_from_file(file_path_json)

    graph = create_graph(graph_filepath)

    outcomes_data = calculate_all_partition_outcomes(
        graph,
        town_populations,
        voter_party1_proportions,
        num_districts,
        max_path_length,
    )

    print("Outcomes data:")
    print(
        f"Party 1 proportion of total votes: {outcomes_data['proportion_party1_total_votes']}"
    )
    print(
        f"Number of possible districts: {outcomes_data['num_possible_districts']}"
    )
    print(f"Number of partitions: {outcomes_data['num_partitions']}")
    print(
        f"Number of partitions favoring party 1: {outcomes_data['partition_party1_wins']}"
    )
    print(
        f"Number of partitions favoring party 2: {outcomes_data['partition_party2_wins']}"
    )
    print(
        f"Number of partitions resulting in a tie: {outcomes_data['partition_ties']}"
    )


def manual_gerrymander():
    """
    Run the gerrymandering scenario and print the outcomes.
    """
    file_name = "gerrymandering_data"  # base name, no .json or .txt

    # read data and build graph once (statewide votes don't change)
    file_path_json = f"data/{file_name}.json"
    (
        graph_filepath,
        num_districts,
        town_populations,
        voter_party1_proportions,
    ) = read_data_from_file(file_path_json)

    graph = create_graph(graph_filepath)

    # calculate statewide vote percentages
    total_votes_party1 = sum(
        town_populations[town] * voter_party1_proportions[town] for town in town_populations
    )
    total_votes = sum(town_populations.values())
    prop_party1 = total_votes_party1 / total_votes
    prop_party2 = 1 - prop_party1

    print(f"For the entire state, Party 1 receives {prop_party1*100:.2f}% of votes, "
          f"Party 2 receives {prop_party2*100:.2f}% of votes.\n")

    # store results for compact and loose partitions
    results = {}

    for max_path_length in [2, 5]:
        outcomes = calculate_all_partition_outcomes(
            graph,
            town_populations,
            voter_party1_proportions,
            num_districts,
            max_path_length,
        )

        num_partitions = outcomes["num_partitions"]
        p1_wins = outcomes["partition_party1_wins"]
        p2_wins = outcomes["partition_party2_wins"]

        key = "Compact" if max_path_length == 2 else "Loose"
        results[key] = {
            "num_partitions": num_partitions,
            "p1_percent": (p1_wins / num_partitions * 100) if num_partitions > 0 else 0,
            "p2_percent": (p2_wins / num_partitions * 100) if num_partitions > 0 else 0,
        }

    # print summary
    for key in ["Compact", "Loose"]:
        print(f"Number of possible combinations of {key.lower()} districts = {results[key]['num_partitions']}")
        print(f"{key} partitions favoring Party 1 = {results[key]['p1_percent']:.2f}%")
        print(f"{key} partitions favoring Party 2 = {results[key]['p2_percent']:.2f}%\n")

    # salamander effect?
    for max_path_length in [2, 3, 4, 5]:
        print(f"\n=== Visualizing partitions for max_path_length = {max_path_length} ===")
        # plot first 3 partitions
        manual_get_all_valid_partitions(file_name, max_path_length=max_path_length, plot_k=3)


if __name__ == "__main__":
    # manually test your functions here

   manual_test_print_files("mini_1", True, True)
   manual_get_all_town_combos("mini_1")
   manual_get_all_valid_districts("mini_1")
   manual_get_all_valid_partitions("mini_1")
   manual_calculate_voting_outcomes("mini_1")
   manual_gerrymander()

    ############################################################
    # gerrymandering
    ############################################################

    # defining a single test graph
    # pass

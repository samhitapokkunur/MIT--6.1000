# standard library
from functools import wraps
import math
import json
import os
import random
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
# Test Variables
############################################################

SEED = 0

############################################################
# Test Cases
############################################################


# Spaces
@testsuite_options(4, 1)
class TestPart3(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="Discrete Box methods do not function as expected",
        error="Discrete Box methods raised an error",
    )
    def test_space_1_discrete_box(self):
        random.seed(SEED)
        high1, high2 = (4, 5)
        box = pset.DiscreteBox((0, 0), (high1, high2))
        points_inside = [
            (random.randint(0, high1), random.randint(0, high2)) for _ in range(10)
        ]
        points_outside = [(high1 + 1, high2), (-1, 0), (0, high2 + 1)]
        for point in points_inside:
            self.assertTrue(
                box.contains(point), f"Point {point} should be inside the box | (class: DiscreteBox)"
            )
        for point in points_outside:
            self.assertFalse(
                box.contains(point), f"Point {point} should be outside the box | (class: DiscreteBox)"
            )

        # test sample method
        for _ in range(100):
            sample = box.sample()
            self.assertTrue(
                box.contains(sample), f"Sample {sample} should be inside the box | (class: DiscreteBox)"
            )

    @case_options(
        1,
        failure="Direction Space 2D methods do not function as expected",
        error="Direction Space 2D methods raised an error",
    )
    def test_space_2_direction_space_2d(self):
        space = pset.DirectionSpace2D()
        valid_directions = ["up", "down", "left", "right"]
        for direction in valid_directions:
            self.assertTrue(
                space.contains(direction), f"Direction {direction} should be valid | (class: DirectionSpace2D)"
            )
        invalid_directions = ["upleft", "downright", "forward", "backward", "", None]
        for direction in invalid_directions:
            self.assertFalse(
                space.contains(direction), f"Direction {direction} should be invalid | (class: DirectionSpace2D)"
            )

        # test sample method
        sampled_directions = set()
        for _ in range(100):
            sample = space.sample()
            self.assertIn(
                sample, valid_directions, f"Sampled direction {sample} should be valid | (class: DirectionSpace2D)"
            )
            sampled_directions.add(sample)

        self.assertSetEqual(
            set(valid_directions),
            sampled_directions,
            "Not all valid directions were sampled | (class: DirectionSpace2D)",
        )


# Single Agent Env
@testsuite_options(4, 2)
class TestPart4(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="TaxiEnv constructor does not initialize fields correctly",
        error="TaxiEnv constructor raised an error",
    )
    def test_taxi_env_1_constructor(self):
        # check space types
        length, width = 5, 6
        env = pset.TaxiEnv(length, width)
        self.assertIsInstance(env.action_space, pset.DirectionSpace2D)
        self.assertIsInstance(env.observation_space, pset.DiscreteBox)

        # check observation boundaries
        in_env_points = [(0, 0), (length, width), (2, 3)]
        out_env_points = [(-1, 0), (length + 1, width), (2, width + 1)]
        for point in in_env_points:
            self.assertTrue(
                env.observation_space.contains(point),
                f"Point {point} should be inside the observation space | (class: TaxiEnv)",
            )
        for point in out_env_points:
            self.assertFalse(
                env.observation_space.contains(point),
                f"Point {point} should be outside the observation space | (class: TaxiEnv)",
            )

        # check uninitialized fields
        for field in ["taxi_location", "passenger_location", "destination_location"]:
            self.assertIsNone(
                getattr(env, field), f"{field} should be None upon initialization | (class: TaxiEnv)"
            )

    @case_options(
        1,
        failure="reset() does not initialize the environment state correctly",
        error="reset() raised an error",
    )
    def test_taxi_env_2_reset(self):
        for field in ["taxi_location", "passenger_location", "destination_location"]:
            env = pset.TaxiEnv(5, 6)
            obs, done = env.reset()
            location = getattr(env, field)
            self.assertEqual(location, obs[field], "should be same as observation | (class: TaxiEnv)")
            self.assertIsNotNone(location, f"{field} should not be None after reset | (class: TaxiEnv)")
            self.assertTrue(
                env.observation_space.contains(location),
                f"{field} {location} should be within the observation space after reset | (class: TaxiEnv)",
            )
        # test if taxi, passenger, destination initialized to be the same, done is True
        for _ in range(100):
            env = pset.TaxiEnv(3, 3)
            obs, done = env.reset()
            if obs["taxi_location"] == obs["passenger_location"] == obs["destination_location"]:
                assert(done)

    @case_options(
        1,
        failure="_get_next_location() does not return the correct next location",
        error="_get_next_location() raised an error",
    )
    def test_taxi_env_3_get_next_location(self):
        env = pset.TaxiEnv(5, 5)
        test_cases = [
            # (current_location, action, expected_next_location)
            ((2, 2), "up", (2, 3)),
            ((2, 2), "down", (2, 1)),
            ((2, 2), "left", (1, 2)),
            ((2, 2), "right", (3, 2)),
            # boundary conditions
            ((0, 0), "down", (0, 0)),
            ((0, 0), "left", (0, 0)),
            ((5, 5), "up", (5, 5)),
            ((5, 5), "right", (5, 5)),
        ]
        for current_location, action, expected_location in test_cases:
            actual_location = env._get_next_location(current_location, action)
            self.assertEqual(
                actual_location,
                expected_location,
                f"From {current_location} taking action {action}, expected {expected_location} but got {actual_location} | (class: TaxiEnv)",
            )

    @case_options(
        1,
        failure="step() does not update the environment state correctly",
        error="step() raised an error",
    )
    def test_taxi_env_4_step_1(self):
        env = pset.TaxiEnv(5, 5)
        env.taxi_location = (0, 0)
        env.passenger_location = (2, 2)
        env.destination_location = (3, 4)

        actions_to_passenger = ["up", "up", "right", "right"]
        for action in actions_to_passenger:
            self.assertFalse(
                env.picked_up_passenger,
                "Passenger should not be picked up en route to passenger location | (class: TaxiEnv)",
            )
            obs, done = env.step(action)
        self.assertTrue(
            env.taxi_location == env.passenger_location,
            f"Taxi should be at passenger location. Got {env.taxi_location} instead of {env.passenger_location} | (class: TaxiEnv)",
        )
        self.assertTrue(
            env.picked_up_passenger,
            "Passenger should be picked up when taxi reaches passenger location | (class: TaxiEnv)",
        )
        self.assertFalse(done)
        self.assertTrue(env.t == 4, "need to update time step | (class: TaxiEnv)")

        actions_to_destination = ["right", "up", "up"]
        for action in actions_to_destination:
            self.assertTrue(
                env.picked_up_passenger,
                "Passenger should have been picked up before heading to destination | (class: TaxiEnv)",
            )
            obs, done = env.step(action)
        self.assertTrue(
            env.taxi_location == env.destination_location,
            "Taxi should be at destination location | (class: TaxiEnv)",
        )
        self.assertTrue(
            done, "Episode should be done when taxi reaches destination with passenger | (class: TaxiEnv)"
        )

    @case_options(
        1,
        failure="step() does not update the environment state correctly",
        error="step() raised an error",
    )
    def test_taxi_env_4_step_2(self):
        env = pset.TaxiEnv(5, 5)
        env.taxi_location = (0, 0)
        env.passenger_location = (2, 2)
        env.destination_location = (3, 4)

        actions = ["up", "up", "up", "right", "right", "up", "right"]
        for action in actions:
            obs, done = env.step(action)
            self.assertFalse(done, "passenger not picked up | (class: TaxiEnv)")
        self.assertEqual(env.t, len(actions), "time step should be updated | (class: TaxiEnv)")

    @case_options(
        1,
        failure="_get_next_location() does not account for ice or boulders correctly",
        error="_get_next_location() raised an error",
    )
    def test_taxi_env_5_frozen_get_next_location(self):
        env = pset.FrozenTaxiEnv(5, 5)
        env.reset()
        env.boulders = set([(0, 1), (4, 0)])
        env.step("up")  # Taxi at (0,0) moves to (0,1) but hits boulder and stays
        self.assertEqual(
            env.taxi_location,
            (0, 0),
            "Taxi should not move into a boulder at (0,1) from (0,0) | (class: FrozenTaxiEnv)",
        )

        env.reset()
        env.boulders = set([(0, 1), (4, 0)])
        env.step("right")
        self.assertEqual(
            env.taxi_location,
            (3, 0),
            "Taxi should slide to (3,0) on ice when moving right from (0,0) when (4,0) is a boulder | (class: FrozenTaxiEnv)",
        )

        env.reset()
        env.boulders = set()
        env.step("right")
        self.assertEqual(
            env.taxi_location,
            (5, 0),
            "Taxi should slide to wall (5,0) on ice when moving right from (0,0) with no boulders | (class: FrozenTaxiEnv)",
        )


# Single Agents
@testsuite_options(4, 1)
class TestPart5(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Random Agent
    @case_options(
        1,
        failure="RandomAgent does not select valid actions",
        error="RandomAgent raised an error",
    )
    def test_single_agent_1_random_agent(self):
        num_trials = 250
        env = pset.TaxiEnv(5, 5)
        agent = pset.RandomAgent(env)
        past_actions = set()
        for _ in range(num_trials):
            action = agent.get_action(None)
            past_actions.add(action)
            self.assertIn(
                action,
                env.action_space.directions,
                f"RandomAgent selected invalid action {action} | (class: RandomAgent)",
            )
        for action in env.action_space.directions:
            self.assertIn(
                action,
                past_actions,
                f"RandomAgent did not select action {action} in {num_trials} trials | (class: RandomAgent)",
            )

    # Greedy Agent
    @case_options(
        1,
        failure="GreedyAgent does not select optimal actions",
        error="GreedyAgent raised an error",
    )
    def test_single_agent_2_greedy_agent(self):
        env = pset.TaxiEnv(5, 5)
        agent = pset.GreedyAgent(env)

        # Test case where taxi is en route to passenger
        env.taxi_location = (1, 1)
        env.passenger_location = (2, 2)
        env.destination_location = (0, 0)
        env.picked_up_passenger = False
        obs = env._get_obs()
        action = agent.get_action(obs)
        self.assertIn(
            action,
            ["up", "right"],
            f"GreedyAgent should move towards destination but selected {action} | (class: GreedyAgent)",
        )

        # Test case where taxi is en route to destination
        env.taxi_location = (5, 5)
        env.passenger_location = (2, 2)
        env.destination_location = (1, 5)
        env.picked_up_passenger = True
        obs = env._get_obs()
        action = agent.get_action(obs)
        self.assertIn(
            action,
            ["left"],
            f"GreedyAgent should move towards passenger but selected {action} | (class: GreedyAgent)",
        )

    # Epsilon Greedy Agent
    @case_options(
        1,
        failure="EpsilonGreedyAgent does not balance exploration and exploitation correctly",
        error="EpsilonGreedyAgent raised an error",
    )
    def test_single_agent_3_epsilon_greedy_agent(self):
        # test 1
        expected_mean, expected_std = 0.8498209999999999, 0.010894079079940635
        num_std = 4
        env = pset.TaxiEnv(5, 5)
        epsilon = 0.2
        agent = pset.EpsilonGreedyAgent(env, epsilon=epsilon)
        num_trials = 500

        env.taxi_location = (0, 0)
        env.passenger_location = (0, 2)
        env.destination_location = (4, 4)
        env.picked_up_passenger = False

        action_counts = {action: 0 for action in env.action_space.directions}
        for _ in range(num_trials):
            action = agent.get_action(env._get_obs())
            action_counts[action] += 1

        greedy_action = "up"
        actual_proportion_greedy_action = action_counts[greedy_action] / num_trials
        self.assertAlmostEqual(
            actual_proportion_greedy_action,
            expected_mean,
            delta=num_std * expected_std,
            msg=f"EpsilonGreedyAgent greedy action proportion {actual_proportion_greedy_action} deviates from expected {expected_mean} by more than {num_std} standard deviations | (class: EpsilonGreedyAgent)",
        )

        # test 2
        expected_mean, expected_std = 0.625787, 0.015752829301430283
        num_std = 4
        env = pset.TaxiEnv(5, 5)
        epsilon = 0.5
        agent = pset.EpsilonGreedyAgent(env, epsilon=epsilon)
        num_trials = 500
        env.taxi_location = (3, 4)
        env.passenger_location = (2, 0)
        env.destination_location = (4, 4)
        env.picked_up_passenger = True

        action_counts = {action: 0 for action in env.action_space.directions}
        for _ in range(num_trials):
            action = agent.get_action(env._get_obs())
            action_counts[action] += 1

        greedy_action = "right"
        actual_proportion_greedy_action = action_counts[greedy_action] / num_trials
        self.assertAlmostEqual(
            actual_proportion_greedy_action,
            expected_mean,
            delta=num_std * expected_std,
            msg=f"EpsilonGreedyAgent greedy action proportion {actual_proportion_greedy_action} deviates from expected {expected_mean} by more than {num_std} standard deviations | (class: EpsilonGreedyAgent)",
        )


# Multi Agent Env
@testsuite_options(4, 1)
class TestPart6(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="reset() does not initialize multi-agent environment state correctly",
        error="reset() raised an error",
    )
    def test_multi_agent_env_1_reset(self):
        agent_counts = {
            pset.RandomPerson: 5,
            pset.CarefulPerson: 2,
            pset.MenacingPerson: 3,
        }
        env = pset.CovidSimulation(agent_counts=agent_counts)
        env.reset()
        # Check agent counts
        self.assertIsNotNone(env.agents, "Agents list should not be None after reset | (class: DiseaseSimulation)")
        self.assertIsNotNone(env.grid_index, "grid index should not be None | (class: DiseaseSimulation)")
        self.assertEqual(
            len(env.agents), 10, "Number of agents after reset should be 10 | (class: DiseaseSimulation)"
        )
        for agent in env.agents:
            self.assertIn(
                type(agent),
                agent_counts,
                f"Agent of type {type(agent)} is not recognized | (class: DiseaseSimulation)",
            )
            agent_counts[type(agent)] -= 1
        for agent_type, count in agent_counts.items():
            self.assertEqual(
                count,
                0,
                f"Number of agents of type {agent_type} should be correct after reset | (class: DiseaseSimulation)",
            )

        # Check agent infos
        self.assertIsNotNone(
            env.infos, "Infos dictionary should not be None after reset | (class: DiseaseSimulation)"
        )
        for agent, info in env.infos.items():
            self.assertIn("position", info, "Agent info should contain 'position' | (class: DiseaseSimulation)")
            self.assertIn(
                "infected", info, "Agent info should contain 'infected' status | (class: DiseaseSimulation)"
            )
            self.assertIn("active", info, "Agent info should contain 'active' status | (class: DiseaseSimulation)")
            position, infected, active = (
                info["position"],
                info["infected"],
                info["active"],
            )
            self.assertTrue(
                env.observation_space.contains(position),
                f"Agent location {position} should be within the observation space | (class: DiseaseSimulation)",
            )
            self.assertTrue(active, "Agent should be active upon reset | (class: DiseaseSimulation)")

    @case_options(
        1,
        failure="get_neighbors_within_radius() does not return correct neighbors",
        error="get_neighbors_within_radius() raised an error",
    )
    def test_multi_agent_env_2_get_neighbors_within_radius_correctness(self):
        env = pset.CovidSimulation(agent_counts={})
        env.reset()
        # Manually add agents at known locations
        env.agents = []
        env.infos = {}
        positions = [(2, 2), (2.6, 2.6), (3, 3), (5, 5), (8, 8)]
        for i, pos in enumerate(positions):
            agent = pset.RandomPerson(env)
            env.agents.append(agent)
            env.infos[agent] = {"position": pos, "infected": False, "active": True}
        env._build_grid_index()

        # Query neighbors within radius 2 of (3,3)
        radius = 2
        query_index = 1
        query_agent, query_point = (
            env.agents[query_index],
            env.infos[env.agents[query_index]]["position"],
        )
        actual_neighbors = env.get_neighbors_within_radius(
            query_agent, query_point, radius
        )

        expected_neighbors = [env.agents[0], env.agents[2]]
        self.assertListEqual(
            actual_neighbors,
            expected_neighbors,
            f"Neighbor agents within radius {radius} of {query_point} do not match expected agents | (class: DiseaseSimulation)",
        )

    @case_options(
        1,
        failure="get_neighbors_within_radius() was too slow",
        error="get_neighbors_within_radius() raised an error",
    )
    def test_multi_agent_env_2_get_neighbors_within_radius_speed(self):
        agent_counts = {
            pset.RandomPerson: 500,
            pset.CarefulPerson: 300,
            pset.MenacingPerson: 200,
        }
        env = pset.CovidSimulation(agent_counts=agent_counts)

        # Time the neighbor query
        import time

        start_time = time.time()
        env.reset()
        for i in range(len(env.agents)):
            _ = env.get_neighbors_within_radius(env.agents[i], (50, 50), 10)
            end_time = time.time()
            elapsed_time = end_time - start_time

        self.assertLess(
            elapsed_time,
            0.05,
            f"get_neighbors_within_radius took too long: {elapsed_time} seconds | (class: DiseaseSimulation)",
        )

    @case_options(
        1,
        failure="_get_next_location() does not return the correct next location",
        error="_get_next_location() raised an error",
    )
    def test_multi_agent_env_3_get_next_location(self):
        env = pset.CovidSimulation(agent_counts={})
        env.reset()
        test_cases = [
            # (current_location, action, expected_next_location)
            ((5, 5), (0.6, 0.8), (5.6, 5.8)),
            ((0, 0), (-1.0, 0.0), (0.0, 0.0)),  # boundary condition
        ]
        for current_location, action, expected_location in test_cases:
            actual_location = env._get_next_location(current_location, action)
            self.assertEqual(
                actual_location,
                expected_location,
                f"From {current_location} taking action {action}, expected {expected_location} but got {actual_location} | (class: DiseaseSimulation)",
            )

    @case_options(
        1,
        failure="step() does not infect correctly",
        error="step() raised an error",
    )
    def test_multi_agent_env_4_step_1(self):
        env = pset.CovidSimulation(agent_counts={})
        env.reset()
        # hard code for test cases
        env.infection_prob = 1.0
        env.recovery_prob = 0.0
        env.death_prob = 0.0
        env.infection_radius = 1.0
        # Manually add agents at known locations
        env.agents = []
        env.infos = {}
        positions = [(2, 2), (2.6, 2.6), (3, 3), (5, 5), (8, 8)]
        for i, pos in enumerate(positions):
            agent = pset.RandomPerson(env)
            env.agents.append(agent)
            infected = False
            if pos == (2, 2):
                infected = True
            env.infos[agent] = {"position": pos, "infected": infected, "active": True}
        env._build_grid_index()
        # test step infection
        env.step({agent: (0.5, 0.5) for agent in env.agents})
        for agent in env.agents[:2]:
            self.assertTrue(env.infos[agent]["infected"], "expected agents to be infected | (class: DiseaseSimulation)")
        for agent in env.agents[2:]:
            self.assertFalse(env.infos[agent]["infected"], "expected agents to be not infected | (class: DiseaseSimulation)")
        self.assertTrue(all([env.infos[agent]["active"] for agent in env.agents]), "expected all agents active | (class: DiseaseSimulation)")

    @case_options(
        1,
        failure="step() does not correctly implement death and recovery",
        error="step() raised an error",
    )
    def test_multi_agent_env_4_step_2(self):
        env = pset.CovidSimulation(agent_counts={})
        env.reset()
        # hard code for test cases
        env.recovery_prob = 1.0
        env.death_prob = 0.8
        num_agents = 1000
        # Manually add infected agents at random positions
        env.infos = {}
        env.agents = []
        for _ in range(num_agents):
            agent = pset.RandomPerson(env)
            env.agents.append(agent)
            env.infos[agent] = {"position": env.action_space.sample(), "infected": True, "active": True}
        env._build_grid_index()
        # test step death
        infos, done = env.step({agent: env.action_space.sample() for agent in env.agents})
        num_dead = sum(1 for agent, info in infos.items() if not info["active"])
        num_infected = sum(1 for agent, info in infos.items() if info["infected"])
        expected_num_dead = env.death_prob * num_agents
        epsilon = 25
        self.assertAlmostEqual(num_dead, expected_num_dead, delta=epsilon, msg=f"expected approximately {expected_num_dead} | (class: DiseaseSimulation)")
        self.assertAlmostEqual(num_infected, 0, delta=epsilon, msg=f"expected approzimately 0 infected agents | (class: DiseaseSimulation)")


# Multi Agents
@testsuite_options(4, 2)
class TestPart7(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        failure="get_nearest_agent_with_filtered_radius does not return correct nearest agent",
        error="get_nearest_agent_with_filtered_radius raised an error",
    )
    def test_person_1_get_nearest_agent_with_filtered_radius(self):
        env = pset.CovidSimulation(agent_counts={})
        careful_agent = pset.CarefulPerson(env)
        random_agent_1 = pset.RandomPerson(env)
        random_agent_2 = pset.RandomPerson(env)
        random_agent_3 = pset.RandomPerson(env)
        random_agent_4 = pset.RandomPerson(env)
        env.agents = [
            careful_agent,
            random_agent_1,
            random_agent_2,
            random_agent_3,
            random_agent_4,
        ]
        env.infos = {
            careful_agent: {"position": (5, 5), "infected": False, "active": True},
            random_agent_1: {  # infected, closest infected
                "position": (4.5, 5.1),
                "infected": True,
                "active": True,
            },
            random_agent_2: {  # infected, farther
                "position": (5, 5.8),
                "infected": True,
                "active": True,
            },
            random_agent_3: {  # healthy, closest overall
                "position": (4.9, 5.1),
                "infected": False,
                "active": True,
            },
            random_agent_4: {  # infected, inactive
                "position": (100, 100),
                "infected": False,
                "active": False,
            },
        }
        env._build_grid_index()
        filter_infected = lambda info: info["infected"]
        filter_healthy = lambda info: not info["infected"]
        filter_inactive = lambda info: not info["active"]

        expected_infected = careful_agent.get_nearest_agent_with_filtered_radius(
            env.infos, 5, filter_infected
        )
        expected_healthy = careful_agent.get_nearest_agent_with_filtered_radius(
            env.infos, 5, filter_healthy
        )
        expected_inactive = careful_agent.get_nearest_agent_with_filtered_radius(
            env.infos, 5, filter_inactive
        )

        self.assertEqual(
            expected_infected,
            random_agent_1,
            "get_nearest_agent_with_filtered_radius did not return the correct nearest infected agent | (class: CarefulPerson)",
        )
        self.assertEqual(
            expected_healthy,
            random_agent_3,
            "get_nearest_agent_with_filtered_radius did not return the correct nearest healthy agent | (class: CarefulPerson)",
        )
        self.assertIsNone(
            expected_inactive,
            "get_nearest_agent_with_filtered_radius should return None when no agents match the filter | (class: CarefulPerson)",
        )

    @case_options(
        1,
        failure="CarefulPerson does not avoid infected neighbors correctly",
        error="CarefulPerson raised an error",
    )
    def test_person_3_careful_person_1(self):
        env = pset.CovidSimulation(agent_counts={})
        careful_agent = pset.CarefulPerson(env)
        random_agent_1 = pset.RandomPerson(env)
        random_agent_2 = pset.RandomPerson(env)
        random_agent_3 = pset.RandomPerson(env)
        env.agents = [careful_agent, random_agent_1, random_agent_2, random_agent_3]
        env.infos = {
            careful_agent: {"position": (5, 5), "infected": False, "active": True},
            random_agent_1: {  # infected, closest infected
                "position": (4.5, 5.1),
                "infected": True,
                "active": True,
            },
            random_agent_2: {  # infected, farther
                "position": (5, 5.8),
                "infected": True,
                "active": True,
            },
            random_agent_3: {  # healthy, closest overall
                "position": (4.9, 5),
                "infected": False,
                "active": True,
            },
        }
        env._build_grid_index()
        dx, dy = careful_agent.get_action(env.infos)
        matches_expected_direction_dx = dx > 0
        matches_expected_direction_dy = dy < 0
        self.assertTrue(
            matches_expected_direction_dx,
            msg=f"CarefulPerson should move away from the closest infected neighbor | (class: CarefulPerson)",
        )
        self.assertTrue(
            matches_expected_direction_dy,
            msg=f"CarefulPerson should move away from the closest infected neighbor | (class: CarefulPerson)",
        )
        self.assertTrue(
            env.action_space.contains((dx, dy)),
            msg="CarefulPerson selected an invalid action | (class: CarefulPerson)",
        )

    @case_options(
        1,
        failure="MenacingPerson does not move towards healthy neighbors correctly when infected",
        error="MenacingPerson raised an error",
    )
    def test_person_2_menacing_person(self):
        env = pset.CovidSimulation(agent_counts={})
        menacing_agent = pset.MenacingPerson(env)
        random_agent_1 = pset.RandomPerson(env)
        random_agent_2 = pset.RandomPerson(env)
        random_agent_3 = pset.RandomPerson(env)
        env.agents = [menacing_agent, random_agent_1, random_agent_2, random_agent_3]
        env.infos = {
            menacing_agent: {"position": (5, 5), "infected": True, "active": True},
            random_agent_1: {  # healthy, closest healthy
                "position": (4.5, 5.1),
                "infected": False,
                "active": True,
            },
            random_agent_2: {  # infected, closest overall
                "position": (5.1, 5),
                "infected": True,
                "active": True,
            },
            random_agent_3: {  # healthy, farther
                "position": (6, 5),
                "infected": False,
                "active": True,
            },
        }
        env._build_grid_index()
        dx, dy = menacing_agent.get_action(env.infos)
        matches_expected_direction_dx = dx < 0
        matches_expected_direction_dy = dy > 0
        self.assertTrue(
            matches_expected_direction_dx,
            msg=f"MenacingPerson should move towards the closest healthy neighbor | (class: MenacingPerson)",
        )
        self.assertTrue(
            matches_expected_direction_dy,
            msg=f"MenacingPerson should move towards the closest healthy neighbor | (class: MenacingPerson)",
        )
        self.assertTrue(
            env.action_space.contains((dx, dy)),
            msg="MenacingPerson selected an invalid action | (class: MenacingPerson)",
        )

    @case_options(
        1,
        failure="CarefulPerson does not avoid healthy neighbors correctly when infected",
        error="CarefulPerson raised an error",
    )
    def test_person_3_careful_person_2(self):
        env = pset.CovidSimulation(agent_counts={})
        careful_agent = pset.CarefulPerson(env)
        random_agent_1 = pset.RandomPerson(env)
        random_agent_2 = pset.RandomPerson(env)
        random_agent_3 = pset.RandomPerson(env)
        env.agents = [careful_agent, random_agent_1, random_agent_2, random_agent_3]
        env.infos = {
            careful_agent: {"position": (5, 5), "infected": True, "active": True},
            random_agent_1: {  # infected, closest overall
                "position": (5.1, 5.1),
                "infected": True,
                "active": True,
            },
            random_agent_2: {  # healthy, closest healthy
                "position": (5.2, 4.8),
                "infected": False,
                "active": True,
            },
            random_agent_3: {  # healthy, farther
                "position": (5, 5.8),
                "infected": False,
                "active": True,
            },
        }
        env._build_grid_index()
        dx, dy = careful_agent.get_action(env.infos)
        matches_expected_direction_dx = dx < 0
        matches_expected_direction_dy = dy > 0
        self.assertTrue(
            matches_expected_direction_dx,
            msg=f"CarefulPerson should move away from the closest healthy neighbor | (class: CarefulPerson)",
        )
        self.assertTrue(
            matches_expected_direction_dy,
            msg=f"CarefulPerson should move away from the closest healthy neighbor | (class: CarefulPerson)",
        )
        self.assertTrue(
            env.action_space.contains((dx, dy)),
            msg="CarefulPerson selected an invalid action | (class: CarefulPerson)",
        )


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


############################################################
# do not include the "__main__" section for test.py in _files/
############################################################

if __name__ == "__main__":
    test_parts = [
        TestPart3,
        TestPart4,
        TestPart5,
        TestPart6,
        TestPart7,
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

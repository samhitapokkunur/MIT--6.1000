"""
6.1000 Fall 2025
Problem Set 5

Please fill out the following info:
Name:
Kerberos:
Approximate time spent (HH:MM):
"""

import random
import copy
import time
from visualization import (
    TaxiEnvVisualizaton,
    FrozenTaxiEnvVisualization,
    DiseaseSimVisualization,
)
from grid_index import GridIndex
import numpy as np
import math


############################################################
# Section 2: Introduction to Agents and Environments
############################################################


def run_single_agent_episode(env, agent, max_num_steps=100, render=True):
    """
    Run a single episode for a single agent environment.
    Params:
        env (Env): the environment to run the episode in
        agent (BaseAgent): the agent to interact with the environment
        max_num_steps (int): maximum number of steps to run the episode for
        render (bool): whether to render the environment at each step
    """
    obs, done = env.reset()
    time_step = 0
    while not done and time_step < max_num_steps:
        action = agent.get_action(obs)
        obs, done = env.step(action)
        time_step += 1
        if render:
            env.render(time_step)
    print(f"Episode finished! {time_step} steps taken.")


def run_multi_agent_episode(env, max_num_steps=100, render=True):
    """
    Run a single episode for a multi-agent environment.
    Params:
        env (MultiAgentEnv): the environment to run the episode in
        max_num_steps (int): maximum number of steps to run the episode for
        render (bool): whether to render the environment at each step
    """
    start_time = time.time()
    obs, done = env.reset()
    time_step = 0
    while not done and time_step < max_num_steps:
        actions = {agent: agent.get_action(obs) for agent in env.get_agents()}
        obs, done = env.step(actions)
        time_step += 1
        if render:
            env.render(time_step)

    end_time = time.time()
    print(f"Episode finished in {end_time - start_time:.2f} seconds!")


############################################################
# Section 3: Spaces
############################################################


# Space abstract class, do not modify
class Space:
    """
    Abstract base class for a space, which defines a set of possible elements.
    """

    def __init__(self):
        pass

    def sample(self):
        """
        Randomly sample an element from this space.
        """
        raise NotImplementedError("Abstract method, should not be implemented")

    def contains(self, element):
        """
        Params:
            element: element to be checked if it is in the space
        Returns:
            True if the element is in the space, False otherwise
        """
        raise NotImplementedError("Abstract method, should not be implemented")


########################################
# 3.1: Box, UnitCircle
########################################


# Staff given Box class, do not modify
class Box(Space):
    """
    Abstract base class for a space, which defines a set of possible elements within
    an n-dimensional box.
    """

    def __init__(self, lows, highs):
        assert len(lows) == len(highs), "Low and high must be of the same dimension"
        self.lows = lows
        self.highs = highs
        self.shape = (len(lows),)

    # NOTE: leave this unimplemented
    def sample(self):
        raise NotImplementedError("sample method not implemented")

    def contains(self, element):
        for e, low, high in zip(element, self.lows, self.highs):
            if e < low or e > high:
                return False
        return True


# Staff given UnitCircle class, do not modify
class UnitCircle(Space):
    """
    A 2D unit circle space, representing all points (x, y) such that x^2 + y^2 <= 1.
    """

    def __init__(self):
        self.epsilon = 1e-3

    def sample(self):
        angle = random.uniform(0, 2 * math.pi)
        return (math.cos(angle), math.sin(angle))

    def contains(self, element):
        x, y = element
        return (
            isinstance(x, float)
            and isinstance(y, float)
            and math.sqrt(x**2 + y**2) <= 1 + self.epsilon
        )


########################################
# 3.2: Continuous, Discrete, Direction
########################################


# Staff given ContinuousBox, do not modify
class ContinuousBox(Box):
    """
    A box in continuous space, where each dimension is a continuous range,
    representing all continuous points x = (x[0], x[1], ..., x[d-1]) such that x[i]
    in the interval [lows[i], highs[i]] for each dimension i.
    """

    def __init__(self, lows, highs):
        super().__init__(lows, highs)

    def sample(self):
        return tuple(
            [random.uniform(low, high) for low, high in zip(self.lows, self.highs)]
        )

    def contains(self, element):
        return super().contains(element) and all(isinstance(e, float) for e in element)


class DiscreteBox(Box):
    """
    A box in discrete space, where each dimension is a discrete range of integers, representing all
    tuples of integer coordinates such that x[i] in {lows[i], lows[i]+1, ..., highs[i]}
    for each dimension i.
    """

    def __init__(self, lows, highs):
        super().__init__(lows, highs)

    def sample(self):
        return tuple(random.randint(low, high) for low, high in zip(self.lows, self.highs))
        raise NotImplementedError("sample method not implemented")

    def contains(self, element):
        if not all(isinstance(e, int) for e in element):
            return False
        return super().contains(element)
        raise NotImplementedError("contains method not implemented")


class DirectionSpace2D(Space):
    """
    A space representing the four cardinal directions.
    """

    def __init__(self):
        self.directions = ("up", "down", "left", "right")

    def sample(self):
        return random.choice(self.directions)
        raise NotImplementedError("sample method not implemented")

    def contains(self, element):
        return isinstance(element, str) and element in self.directions
        raise NotImplementedError("contains method not implemented")


############################################################
# Section 4: Single Agent Simulations
############################################################


class Env:
    def __init__(self):
        self.action_space = None
        self.observation_space = None

    def reset(self):
        """
        Reset the environment to an initial state.
        Returns:
            observation (object): the initial observation of the space.
            done (bool): whether the episode is done.
        """
        raise NotImplementedError("Abstract method, should not be implemented")

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Params:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            done (bool): whether the episode has ended
        """
        raise NotImplementedError("Abstract method, should not be implemented")


########################################
# Section 4.1-4.4: Taxi Environment
########################################


class TaxiEnv(Env):
    # 4.1 Implement TaxiEnv Constructor
    def __init__(self, width, height):
        """
        Initialize the Taxi Environment.
        """
        super().__init__()
        assert width > 0 and height > 0, "Width and height must be positive integers"

        # Dimensions
        self.width = width
        self.height = height

        # Spaces
        self.action_space = DirectionSpace2D()
        self.observation_space = DiscreteBox([0, 0], [width - 1, height - 1])

        # State variables
        self.taxi_location = None
        self.passenger_location = None
        self.destination_location = None
        self.picked_up_passenger = False
        self.t = 0

        # Optional visualization (only if provided)
        if 'TaxiEnvVisualizaton' in globals():
            self.vis = TaxiEnvVisualizaton(self)

    # Staff given, do not modify
    def _get_obs(self):
        """
        Outputs the current observation of the environment.

        Returns:
            observation (dict): current observation of the environment
        """
        return {
            "taxi_location": self.taxi_location,
            "passenger_location": self.passenger_location,
            "destination_location": self.destination_location,
            "picked_up_passenger": self.picked_up_passenger,
        }

    # 4.2 Implement TaxiEnv reset method
    def reset(self):
        """
        Reset the environment to a random initial state.

        Returns:
            observation (dict): the initial observation of the environment
            done (bool): whether the episode is done
        """
        # Randomly choose taxi, passenger, and destination positions
        self.taxi_location = self.observation_space.sample()
        self.passenger_location = self.observation_space.sample()
        self.destination_location = self.observation_space.sample()

        # Ensure passenger and destination are different
        while self.destination_location == self.passenger_location:
            self.destination_location = self.observation_space.sample()

        # Reset flags and timestep
        self.picked_up_passenger = False
        self.t = 0

        done = False  # always false at start
        return self._get_obs(), done


        raise NotImplementedError("reset method not implemented")

    # 4.3 Implement TaxiEnv _get_next_location method
    def _get_next_location(self, location, action):
        """
        Get the next location given the current location and an action.
        Params:
            location (tuple): current (x, y) location
            action (str): action to be taken
        Returns:
            new_location (tuple): new (x, y) location after taking the action
        """

        x, y = location

        if action == "up":
            new_location = (x, y-1)
        elif action == "down":
            new_location = (x, y+1)
        elif action == "left":
            new_location = (x - 1, y)
        elif action == "right":
            new_location = (x + 1, y)
        else:
            return location

        if self.observation_space.contains(new_location):
            return new_location
        else:
            return location
        raise NotImplementedError("_get_next_location method not implemented")

    # 4.4 Implement TaxiEnv step method
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Params:
            action (str): action to be taken
        Returns:
            observation (object): agent's observation of the current environment
            done (bool): whether the episode has ended
        """

        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Move taxi according to action
        self.taxi_location = self._get_next_location(self.taxi_location, action)

        # Pick up passenger if on same cell and not yet picked up
        if not self.picked_up_passenger and self.taxi_location == self.passenger_location:
            self.picked_up_passenger = True

        # If passenger is picked up, they move with the taxi
        if self.picked_up_passenger:
            self.passenger_location = self.taxi_location

        # Advance timestep
        self.t += 1

        # Check if destination reached (episode done)
        done = self.picked_up_passenger and (self.taxi_location == self.destination_location)

        return self._get_obs(), done

        raise NotImplementedError("step method not implemented")

    def render(self, time_step):
        """
        Render the current state of the environment.
        """
        print("====Current State====:")
        print(f"Taxi Location: {self.taxi_location}")
        print(f"Passenger Location: {self.passenger_location}")
        print(f"Destination: {self.destination_location}")

        self.vis.render(time_step)


########################################
# Section 4.5: Frozen Taxi Environment
########################################


class FrozenTaxiEnv(TaxiEnv):
    def __init__(self, width, height):
        """
        Initialize the FrozenTaxiEnv with the given width and height.
        The floor is slippery, causing the taxi to slide until it hits a boulder or wall.
        Boulders are placed randomly while ensuring they do not block the path.
        """
        super().__init__(width, height)
        assert width >= 2 and height >= 2, "Width and height must be at least 2"
        self.num_boulders = (height + width) / 2

        # for visualization - DO NOT CHANGE
        self.vis = FrozenTaxiEnvVisualization(self)

    def _distance(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def reset(self):
        """
        Reset the environment to a randomized initial state. Initial taxi, passenger,
        and destination positions are fixed.
        """
        super().reset()
        self.taxi_location = (
            self.observation_space.lows[0],
            self.observation_space.lows[1],
        )  # Fixed start position
        self.passenger_location = (
            self.observation_space.highs[0],
            self.observation_space.highs[1],
        )  # Fixed passenger position
        self.destination_location = (
            self.observation_space.lows[0],
            self.observation_space.highs[1],
        )  # Fixed destination

        self.boulders = set()
        i = 0
        number_of_attempts = 0
        while i < self.num_boulders:
            boulder = self.observation_space.sample()
            if (
                boulder != self.taxi_location
                and boulder != self.passenger_location
                and boulder != self.destination_location
                and self._distance(boulder, self.taxi_location) > 2
                and self._distance(boulder, self.passenger_location) > 2
                and self._distance(boulder, self.destination_location) > 2
            ):
                self.boulders.add(boulder)
                i += 1
            if number_of_attempts > 1000:
                raise Exception("Could not place boulders without blocking the path")
            number_of_attempts += 1

        self.picked_up_passenger = self.taxi_location == self.passenger_location
        return (
            self._get_obs(),
            self.picked_up_passenger
            and self.taxi_location == self.destination_location,
        )

    # 4.5 Implement FrozenTaxiEnv _get_next_location method
    def _get_next_location(self, location, action):
        """
        Get the next location given the current location and an action, given a slippery surface with boulders.

        Params:
            location (tuple): current (x, y) location
            action (str): action to be taken ('up', 'down', 'left', 'right')

        Returns:
            new_location (tuple): new (x, y) location after taking the action
        """
        x, y = location
        if action == "up":
            dx, dy = 0, -1
        elif action == "down":
            dx, dy = 0, 1
        elif action == "left":
            dx, dy = -1, 0
        elif action == "right":
            dx, dy = 1, 0
        else:
            return location

        next_loc = (x + dx, y + dy)
        # Continue sliding as long as the next location is within bounds AND is not a boulder
        while self.observation_space.contains(next_loc) and next_loc not in self.boulders:
            x, y = next_loc
            next_loc = (x + dx, y + dy)

        # The taxi stops at the last valid (x, y) before hitting a wall or boulder.
        return (x,y)
        # Original code had: raise NotImplementedError("Method not implemented") - REMOVED!
        #class FrozenTaxiEnv(TaxiEnv):
    # ... (other methods of FrozenTaxiEnv)

    # 4.5 Implement FrozenTaxiEnv _get_next_location method
    def _get_next_location(self, location, action):
        """
        Get the next location given the current location and an action, given a slippery surface with boulders.

        Params:
            location (tuple): current (x, y) location
            action (str): action to be taken ('up', 'down', 'left', 'right')

        Returns:
            new_location (tuple): new (x, y) location after taking the action
        """
        x, y = location
        if action == "up":
            dx, dy = 0, -1
        elif action == "down":
            dx, dy = 0, 1
        elif action == "left":
            dx, dy = -1, 0
        elif action == "right":
            dx, dy = 1, 0
        else:
            return location

        next_loc = (x + dx, y + dy)
        # Continue sliding as long as the next location is within bounds AND is not a boulder
        while self.observation_space.contains(next_loc) and next_loc not in self.boulders:
            x, y = next_loc
            next_loc = (x + dx, y + dy)

        # The taxi stops at the last valid (x, y) before hitting a wall or boulder.
        return (x,y)
        # Original code had: raise NotImplementedError("Method not implemented") - REMOVED!

    def render(self, time_step):
        print("====Current State====:")
        print(f"Taxi Location: {self.taxi_location}")
        print(f"Passenger Location: {self.passenger_location}")
        print(f"Destination: {self.destination_location}")

        self.vis.render(time_step)


############################################################
# Section 5: Single Agent Interfaces
############################################################


# Base Agent class, do not modify
class BaseAgent:
    def __init__(self, env):
        """
        Initialize the agent with the given environment.
        """
        self.env = env

    def get_action(self, observation):
        """
        Given an observation, output an action to take in the environment.
        """
        raise NotImplementedError("get_action method implemented in sub-classes")


# 5.1 Implement RandomAgent class
class RandomAgent(BaseAgent):
    def __init__(self, env):
        """
        Initialize the agent with the given environment.
        """
        super().__init__(env)
        #raise NotImplementedError("__init__ method not implemented")

    def get_action(self, observation):
        """
        Return a random action from the environment's action space.
        """
        return self.env.action_space.sample()
        #raise NotImplementedError("get_action method not implemented")


# 5.2 Implement GreedyAgent class
class GreedyAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def get_action(self, observation):
        taxi_loc = observation["taxi_location"]
        picked_up = observation["picked_up_passenger"]
        # Target is passenger if not picked up, destination if picked up
        target = observation["destination_location"] if picked_up else observation["passenger_location"]

        best_action = None
        best_dist = float("inf")
        action_order = ["up", "down", "left", "right"]  # deterministic tie-breaking (favors 'up')

        for action in action_order:
            # Use the environment's specific _get_next_location (handles standard or sliding movement)
            next_loc = self.env._get_next_location(taxi_loc, action)
            # Calculate Manhattan distance (L1 distance)
            dist = abs(next_loc[0] - target[0]) + abs(next_loc[1] - target[1])

            # Update only if the distance is strictly better.
            # This ensures the agent favors the actions earlier in the action_order list if distances are equal (tie-breaker).
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action

class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, env, epsilon=0.1):
        """
        Initialize the EpsilonGreedyAgent with environment and exploration rate epsilon.
        """
        super().__init__(env)
        self.epsilon = epsilon

    def get_action(self, observation):
        """
        With probability epsilon choose a random action, otherwise choose greedy action.
        """
        # Ensure self.epsilon is used here! The error suggests a hardcoded 0.95 was used locally.
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # explore (Random Agent's action)
        return super().get_action(observation)     # exploit (Greedy Agent's action)

############################################################
# Section 6: Disease Simulations and Multi-Agent Environments
############################################################


# Base MultiAgentEnv class, do not modify
class MultiAgentEnv:
    """
    Initialization parameters:
    """

    def __init__(self, agent_counts: dict):
        self.agent_counts = agent_counts
        self.observation_space = None
        self.action_space = None

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            infos (dict): initial information for each agent
            done (bool): whether the simulation is done
        """
        raise NotImplementedError("Abstract method, should not be implemented")

    def step(self, actions):
        """
        Step the environment with the given actions.

        Params:
            actions (dict): dictionary mapping agents to their actions

        Returns:
            infos (dict): updated information for each agent
            done (bool): whether the simulation is done
        """
        raise NotImplementedError("Abstract method, should not be implemented")


########################################
# Section 6.1-6.4: Disease Simulation Env
########################################


class DiseaseSimulation(MultiAgentEnv):
    def __init__(self, agent_counts, disease_parameters):
        """
        Initialize the disease simulation environment.

        Disease simulation uses the following disease parameters:
        - starting_infection_prob (float): probability an agent starts infected
        - infection_prob (float): probability of infection upon contact
        - recovery_prob (float): probability of recovery for infected agents
        - death_prob (float): probability of death for infected agents
        - infection_radius (float): radius within which infection can spread
        """
        super().__init__(agent_counts)
        self.observation_space = ContinuousBox([0, 0], [100, 100])
        self.observation_dim = 2
        self.action_space = UnitCircle()

        self.starting_infection_prob = disease_parameters.get("starting_infection_prob")
        self.infection_prob = disease_parameters.get("infection_prob")
        self.recovery_prob = disease_parameters.get("recovery_prob")
        self.death_prob = disease_parameters.get("death_prob")
        self.infection_radius = disease_parameters.get("infection_radius")

        # for visualization
        self.width = 100
        self.height = 100
        self.vis = DiseaseSimVisualization(self)

    def get_distance(self, loc1, loc2):
        return sum([(loc1[i] - loc2[i]) ** 2 for i in range(len(loc1))]) ** 0.5

    def get_agents(self):
        return self.agents

    # 6.1 Implement reset method & _build_grid_index method
    def reset(self):
        """
        Reset the environment to an initial state. Each agent is placed randomly in the environment.
        All agents start active, and start infected with probability self.starting_infection_prob.
        Builds spatial index for efficient neighbor querying.

        Returns:
            infos (dict): initial information for each agent
            done (bool): whether the simulation is done
        """
        raise NotImplementedError("reset method not implemented")

    def _build_grid_index(self):
        """
        Initializes self.grid_index to a new GridIndex based on current agent positions.
        """
        raise NotImplementedError("_build_grid_index method not implemented")

    # 6.2 Implement get_neighbors_within_radius method
    def get_neighbors_within_radius(self, agent, position, radius):
        """
        Query neighboring agents within a certain radius, excluding the agent itself.

        Returns
        - neighbors (list): list of neighboring agents within the specified radius
        """
        raise NotImplementedError("get_neighbors_within_radius method not implemented")

    # 6.3 Implement _get_next_location method
    def _get_next_location(self, location, action):
        """
        Compute the next location given the current location and action.
        """
        raise NotImplementedError("_get_next_location method not implemented")

    def get_env_stats(self):
        """
        Get statistics about the current state of the environment.

        Returns
        - stats (dict): dictionary containing statistics such as number of infected agents, number of active agents, etc.
        """
        num_infected = 0
        num_active = 0
        for info in self.infos.values():
            if info["infected"]:
                num_infected += 1
            if info["active"]:
                num_active += 1
        return {
            "num_infected": num_infected,
            "num_active": num_active,
            "total_agents": len(self.agents),
        }

    # 6.4 Implement step method
    def step(self, actions):
        """
        Perform a simulation step given the actions of all agents. Agents first move,
        then infection and active status is updated based on proximity. Rebuild spatial index
        after all updates.

        Returns:
            infos (dict): updated information for each agent
            done (bool): whether the simulation is done
        """
        new_infos = {agent: info.copy() for agent, info in self.infos.items()}

        # update infection status
        for agent in self.agents:
            position, infected, active = (
                self.infos[agent]["position"],
                self.infos[agent]["infected"],
                self.infos[agent]["active"],
            )

            if not active:
                continue

            if not infected:
                # infection from other agents
                for other_agent in self.get_neighbors_within_radius(
                    agent, position, self.infection_radius
                ):
                    ...  # TODO : Complete infection logic

            if infected:
                # death
                ...  # TODO : Complete death logic

                # natural recovery
                ...  # TODO : Complete recovery logic

        # update location
        for agent, action in actions.items():
            assert self.action_space.contains(action), "Invalid action"
            prev_position = self.infos[agent]["position"]
            if not new_infos[agent]["active"]:
                continue
            new_position = self._get_next_location(prev_position, action)
            new_infos[agent]["position"] = new_position

        self.infos = new_infos
        done = all(not info["active"] for info in self.infos.values())

        self._build_grid_index()
        return self.infos, done

    def render(self, time_step):
        print("====Current State====:")
        stats = self.get_env_stats()
        print(f"Number of Agents Infected: {stats['num_infected']}")
        print(f"Number of Agents Active: {stats['num_active']}")
        print(f"Number of Agents: {stats['total_agents']}")

        self.vis.render(time_step)


class CovidSimulation(DiseaseSimulation):
    def __init__(self, agent_counts):
        infection_parameters = {
            "starting_infection_prob": 0.2,
            "infection_prob": 0.2,
            "recovery_prob": 0.05,
            "death_prob": 0.005,
            "infection_radius": 4,
        }
        super().__init__(agent_counts, infection_parameters)


class EbolaSimulation(DiseaseSimulation):
    def __init__(self, agent_counts):
        infection_parameters = {
            "starting_infection_prob": 0.2,
            "infection_prob": 0.9,
            "recovery_prob": 0.05,
            "death_prob": 0.5,
            "infection_radius": 2,
        }
        super().__init__(agent_counts, infection_parameters)


########################################
# Section 6.5-6.8: Multi Agent Interfaces
########################################


# Base Person class
class Person:
    person_id = 0

    # Do not modify, base constructor
    def __init__(self, env):
        self.env = env

        self.id = Person.person_id
        Person.person_id += 1

    # Do not modify, use id for equality
    def __eq__(self, other):
        return isinstance(other, Person) and self.id == other.id

    # Do not modify, use id for hashing
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Person-{self.id}"

    # Do not modify, abstract method
    def get_action(self, observation):
        raise NotImplementedError("Do not implement, should be overridden in subclass")

    # Do not modify, abstract method
    def get_type(self):
        raise NotImplementedError("Do not implement, should be overridden in subclass")

    # 6.5 Implement get_nearest_agent_with_filtered_radius method
    def get_nearest_agent_with_filtered_radius(self, observation, radius, filter_fn):
        """
        Get the nearest agent within a certain radius that satisfies the filter function. Filter function
        takes in an one agent's info dictionary and returns True if the the filter condition is satisfied.
        """
        raise NotImplementedError("get_nearest_agent_with_filtered_radius not implemented")


# Staff given RandomPerson class, do not modify
class RandomPerson(Person):
    def __init__(self, env):
        super().__init__(env)

    def get_action(self, observation):
        """
        Return a random action from the action space.
        """
        return self.env.action_space.sample()

    def get_type(self):
        return "random"


# 6.6 Implement MenacingPerson class
class MenacingPerson(Person):
    def __init__(self, env):
        super().__init__(env)
        self.awareness_radius = env.infection_radius * 2
        self.healthy_filter = (
            lambda agent_info: agent_info["infected"] == False
            and agent_info["active"] == True
        )

    def get_action(self, observation):
        """
        Get an action that tries to approach healthy individuals if infected,
        or move randomly if not infected.
        """
        raise NotImplementedError("get_action method not implemented")

    def get_type(self):
        return "menacing"


# 6.7 Implement CarefulPerson class
class CarefulPerson(Person):
    def __init__(self, env):
        super().__init__(env)
        self.awareness_radius = env.infection_radius * 2
        self.infected_filter = (
            lambda agent_info: agent_info["infected"] == True
            and agent_info["active"] == True
        )
        self.healthy_filter = (
            lambda agent_info: agent_info["infected"] == False
            and agent_info["active"] == True
        )

    def get_action(self, observation):
        """
        Get an action that tries to avoid infected individuals if not infected,
        or avoid healthy individuals if infected.
        """
        raise NotImplementedError("get_action method not implemented")

    def get_type(self):
        return "careful"


########################################
# Section 5.3: DIY Research
########################################

if __name__ == "__main__":
    pass

    # Taxi Env
    # env = TaxiEnv(12, 8)
    # agent = RandomAgent(env)
    # run_single_agent_episode(env, agent)

    # Frozen Taxi Env
    # frozen_env = FrozenTaxiEnv(5, 5)
    # frozen_agent = EpsilonGreedyAgent(frozen_env, epsilon=0.75)
    # run_single_agent_episode(frozen_env, frozen_agent)

    # Disease Simulation
    # agent_counts = {
    #     RandomPerson: 100,
    #     # CarefulPerson: 100,
    #     # MenacingPerson: 100,
    # }
    # disease_env = CovidSimulation(agent_counts)
    # run_multi_agent_episode(disease_env, max_num_steps=100, render=True)

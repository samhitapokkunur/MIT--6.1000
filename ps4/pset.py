import random
from helper import produce_animation
import matplotlib.pyplot as plt

############################################################
# simulating ideal MBTA train lines
############################################################
"""
Context: Trains are traveling on a circular track in a counter clockwise direction
"""

# 2.1 helper functions for train simulation on a circular track

def get_distance(loc1, loc2, track_length):
    """
    Calculates the distance in the counter clockwise direction from loc1 to loc2.
      on a circular track of length track_length.

    Parameters:
        loc1 (float): position of point 1 on the track
        loc2 (float): position of point 2 on the track
        track_length (float): length of the track

    Returns:
        numerical distance from loc1 to loc2
    """
    # simple ccw distance computation
    if loc2 > loc1:
        return loc2 - loc1
    elif loc2 == loc1:
        return 0
    else:
        return track_length - (loc1 - loc2)


def will_pass_stop(current_location, next_location, stop_location, track_length):
    """
    Determines if a train moving from current_location to next_location would have
    to pass through stop_location on a circular track of length track_length.

    Parameters:
        current_location (float): current position of the train on the track
        next_location (float): next position of the train on the track
        stop_location (float): position of the stop on the track
        track_length (float): length of the track

    Returns:
        True if the train will pass stop_location, False otherwise.
    """
    # simple ccw check if stop is between current and next
    if current_location < next_location:
        return current_location < stop_location < next_location
    elif current_location > next_location:
        return stop_location < next_location or stop_location > current_location
    return False


# 2.2 ideal simulation
def simulate_trains_ideal(stop_locations, speed, num_steps, track_length):
    """
    Run one MBTA simulation for num_steps steps, with one train starting at each
    initial stop location. Construct a history of train locations.

    When arriving or starting at a stop, trains wait for 2 time steps before continuing.

    Params:
    - stop_locations (list): list of locations where stops are located
    - speed (float): standard speed of each train (miles per time step)
    - num_steps (int): global number of time steps to run
    - track_length (float): length of circular track

    Returns:
    - location_history (list): a nested list of shape (num_steps+1, num_trains),
      where location_history[t][i] is the location of train i at time step t.
      location_history[0] is a list of the initial locations of all trains.
    """
    # number of trains on track
    num_trains = len(stop_locations)
    # store all train positions per timestep
    location_history = [list(stop_locations)]
    # current position of each train
    current_locations = list(stop_locations)
    # small tolerance to avoid floating point rounding errors
    EPSILON = 1e-9

    # 0 = moving, 1/2 = waiting at stop
    train_states = [1] * num_trains
    # sorted stop locations for proper order
    stops_sorted = sorted(stop_locations)
    # determine next stop for each train
    next_stop_indices = [0] * num_trains
    for i in range(num_trains):
        try:
            idx = stops_sorted.index(stop_locations[i])
            next_stop_indices[i] = (idx + 1) % len(stops_sorted)
        except ValueError:
            pass

    # loop for all timesteps
    for t in range(num_steps):
        new_locations = [0.0] * num_trains
        i = 0
        while i < num_trains:
            loc = current_locations[i]
            state = train_states[i]
            next_stop = stops_sorted[next_stop_indices[i]]

            if state in [1, 2]:
                # train is waiting at stop
                new_locations[i] = loc
                train_states[i] = state + 1 if state == 1 else 0
            elif state == 0:
                # train is moving normally
                distance_to_travel = speed
                if next_stop > loc:
                    distance_needed = next_stop - loc
                else:
                    distance_needed = track_length - loc + next_stop

                # if train can reach stop, mark arrival
                if distance_to_travel >= distance_needed - EPSILON:
                    new_locations[i] = next_stop
                    train_states[i] = 1
                    next_stop_indices[i] = (next_stop_indices[i] + 1) % len(stops_sorted)
                else:
                    # otherwise, just move along
                    new_locations[i] = (loc + distance_to_travel) % track_length
                    train_states[i] = 0
            i += 1

        current_locations = new_locations
        location_history.append(new_locations)

    return location_history


# 2.3 monte carlo helpers
def compute_mean(data):
    """
    Compute the average of a list of numbers.

    Parameters:
        data: list of numbers

    Returns: average of the numbers in data
    """
    return sum(data) / len(data)


def compute_standard_deviation(data):
    """
    Compute the standard deviation of a list of numbers.

    Parameters:
        data: list of numbers

    Returns: standard deviation of the numbers in data
    """
    N = len(data)
    mean = sum(data) / N
    return (sum((x - mean) ** 2 for x in data) / N) ** 0.5


# 2.4 intertrain wait stats
def get_intertrain_stats(location_history, stop_locations):
    """
    Compute the average and standard deviation of intertrain wait times across stops.
    """
    # list to store wait times between trains at stops
    wait_times = []
    num_timesteps = len(location_history)
    if num_timesteps < 2:
        return (0, 0)
    num_trains = len(location_history[0])

    # loop through stops to calculate intervals
    for stop in stop_locations:
        for t in range(num_timesteps - 1):
            departed = any(location_history[t][i] == stop and location_history[t + 1][i] != stop for i in range(num_trains))
            if not departed:
                continue
            for future_t in range(t + 1, num_timesteps):
                arrived = any(location_history[future_t][i] == stop for i in range(num_trains))
                if arrived:
                    wait_times.append(future_t - t)
                    break

    if not wait_times:
        return (0, 0)

    return compute_mean(wait_times), compute_standard_deviation(wait_times)


# 2.5 monte carlo simulation
def run_monte_carlo_ideal(stop_locations, speed, num_steps, track_length, num_trials):
    """
    Run multiple trials of the ideal train simulation and compute average metrics.
    """
    avg_waits = []
    example_history = None

    trial = 0
    while trial < num_trials:
        # run one simulation
        loc_history = simulate_trains_ideal(stop_locations, speed, num_steps, track_length)
        avg_wait, std_wait = get_intertrain_stats(loc_history, stop_locations)
        avg_waits.append(avg_wait)
        if example_history is None:
            example_history = loc_history
        trial += 1

   #if not avg_waits:
    #    return (0, 0, None)

    return compute_mean(avg_waits), compute_standard_deviation(avg_waits), example_history


############################################################
# simulating trains in realistic environments
############################################################

def apply_track_halt(speed, p):
    """
    With probability p, the train halts (speed becomes 0). Otherwise, speed remains unchanged.
    """
    #stops train w probability p
    # random halt chance
    if random.random() < p:
        return 0.0
    return speed


def apply_track_gaussian_slow(speed, sigma):
    """
    The speed is reduced by a random amount drawn from a Gaussian distribution with mean 0 and standard deviation sigma.
    """
    # slows train by random amt
    # random slowdown from gaussian distribution
    slowdown = abs(random.gauss(0, sigma))
    new_speed = max(0.0, speed - slowdown)
    return new_speed


def apply_track_uniform_slow(speed, param):
    """
    The speed is reduced to a random value drawn uniformly from the interval [0, speed].
    """
    # slows train by random amount - uniformly
    # random slowdown between 0 and current speed
    new_speed = random.uniform(0, speed)
    if new_speed < 0:
        new_speed = 0.0
    return new_speed


def simulate_trains_realistic(
    stop_locations,
    original_speed,
    num_steps,
    track_length,
    track_params=None,
    track_slowdown_fns=None,
    train_params=None,
    train_slowdown_fns=None,
):
    """
    Run one realistic MBTA simulation for num_steps steps, with one train starting at each location.
    Train may also halt or slow down while at a stop.
    """
    # set default empty lists for optional parameters
    if track_params is None:
        track_params = []
    if track_slowdown_fns is None:
        track_slowdown_fns = []
    if train_params is None:
        train_params = []
    if train_slowdown_fns is None:
        train_slowdown_fns = []

    # determine how many trains to simulate
    num_trains = len(stop_locations)
    if num_trains == 0 or num_steps <= 0:
        return []

    # location_history keeps track of all train positions over time
    location_history = [list(stop_locations)]

    # current_locations stores where each train is at the current time step
    current_locations = list(stop_locations)

    # small constant for floating-point comparisons - was getting errors b4
    EPSILON = 1e-9

    # train_states indicates whether each train is:
    # 0 = moving, 1 = waiting (just arrived), 2 = staying one more step before moving
    train_states = [1] * num_trains

    # sorted list of stops (used to find next stop ahead)
    stops_sorted = sorted(stop_locations)

    # next_stop_indices tracks the index of the next stop for each train
    next_stop_indices = [(stops_sorted.index(loc) + 1) % len(stops_sorted) for loc in stop_locations]

    # start the simulation clock
    t = 0
    while t < num_steps:
        # new_locations will hold the updated positions for all trains
        new_locations = [0.0] * num_trains

        # start each train at its original speed before applying slowdowns
        current_speeds = [original_speed] * num_trains

        # apply train-based slowdown functions
        i = 0
        while i < num_trains:
            speed = original_speed
            j = 0
            # apply each slowdown function in order
            while j < len(train_slowdown_fns):
                param = train_params[j] if j < len(train_params) else None
                # each slowdown_fn takes (speed, current_locations, train_index, track_length, param)
                speed = train_slowdown_fns[j](speed, current_locations, i, track_length, param)
                j += 1
            # store the resulting speed (no negative speeds allowed)
            current_speeds[i] = max(speed, 0.0)
            i += 1

        # apply track-based slowdown functions
        i = 0
        while i < num_trains:
            speed = current_speeds[i]
            j = 0
            # each track slowdown affects speed based on global track conditions
            while j < len(track_slowdown_fns):
                param = track_params[j] if j < len(track_params) else None
                speed = track_slowdown_fns[j](speed, param)
                j += 1
            current_speeds[i] = max(speed, 0.0)
            i += 1

        # update each train’s position and state for the next step
        i = 0
        while i < num_trains:
            current_loc = current_locations[i]
            current_state = train_states[i]
            next_stop_loc = stops_sorted[next_stop_indices[i]]
            speed = current_speeds[i]

            if current_state in [1, 2]:
                # train is currently stopped at a station
                new_locations[i] = current_loc
                # increment the wait state (1 -> 2 -> 0 = ready to move)
                train_states[i] = current_state + 1 if current_state == 1 else 0

            elif current_state == 0:
                # train is moving along the track
                # calculate distance to next stop and next train ahead
                distance_to_stop = get_distance(current_loc, next_stop_loc, track_length)
                distance_to_train = _distance_to_next_train_ahead(current_locations, i, track_length)

                # the train can only move as far as the closest obstacle
                distance_allowed = min(distance_to_stop, distance_to_train)

                # check if the train will reach a stop or another train
                if speed >= distance_allowed - EPSILON:
                    if distance_to_stop <= distance_to_train:
                        # arrives at stop: set position to stop and enter wait state
                        new_locations[i] = next_stop_loc
                        train_states[i] = 1
                        # update next stop index to the next one around the loop
                        next_stop_indices[i] = (next_stop_indices[i] + 1) % len(stops_sorted)
                    else:
                        # train would collide with the one ahead, so stop in place
                        new_locations[i] = current_loc
                        train_states[i] = 0
                else:
                    # move normally by speed distance
                    new_locations[i] = (current_loc + speed) % track_length
                    train_states[i] = 0
            i += 1

        # update positions for next iteration
        current_locations = new_locations

        # record this step’s positions into history
        location_history.append(new_locations)

        # advance the simulation clock
        t += 1

    # return the full history of train positions over all time steps
    return location_history


def run_monte_carlo_realistic(
    stop_locations,
    speed,
    num_steps,
    track_length,
    num_trials,
    track_params=None,
    track_slowdown_fns=None,
    train_params=None,
    train_slowdown_fns=None,
):
    """
    Run multiple trials of the realistic MBTA simulation and compute average intertrain wait times.
    """
    avg_waits = []
    example_history = None

    trial_index = 0
    while trial_index < num_trials:
        # run one trial
        loc_history = simulate_trains_realistic(
            stop_locations, speed, num_steps, track_length,
            track_params, track_slowdown_fns,
            train_params, train_slowdown_fns
        )
        avg_wait, _ = get_intertrain_stats(loc_history, stop_locations)
        avg_waits.append(avg_wait)
        if example_history is None:
            example_history = loc_history
        trial_index += 1

    # handle empty case
    if not avg_waits:
        return 0, 0, None

    mean_wait = compute_mean(avg_waits)
    std_wait = compute_standard_deviation(avg_waits)
    return mean_wait, std_wait, example_history


############################################################
# simulating strategic trains (self-adjusted slowdowns)
############################################################

def _distance_to_next_train_ahead(train_locations, i, track_length):
    """
    Returns the counter-clockwise distance from train i to the closest train ahead.
    This ensures that even if the array indices are out of order, we find the
    *spatially* nearest train ahead on the circular track.
    """
    # helper function added by me to compute spatial distance between trains - reduced error in slowdown monte_carlo_realistic tests
    # finds smallest positive ccw distance to next train
    my_pos = train_locations[i]
    min_distance = track_length
    j = 0
    while j < len(train_locations):
        if j != i:
            d = get_distance(my_pos, train_locations[j], track_length)
            if 0 < d < min_distance:
                min_distance = d
        j += 1
    return min_distance


def apply_train_no_pass(speed, train_locations, train_index, track_length, param):
    """
    If the train would pass another train at its current speed, it slows down to 0.
    Otherwise, speed remains unchanged.
    """
    # prevent passing by limiting speed to distance to next train
    distance = _distance_to_next_train_ahead(train_locations, train_index, track_length)
    return min(speed, distance)


def apply_train_smart_slowdown(speed, train_locations, train_index, track_length, slowing_distance_threshold):
    """
    If the closest train ahead is within slowing_distance_threshold, reduce speed
    proportionally to avoid getting too close. Otherwise, speed remains unchanged.
    """
    # handle invalid threshold
    if slowing_distance_threshold is None or slowing_distance_threshold <= 0:
        return speed

    # compute distance to next train ahead
    distance = _distance_to_next_train_ahead(train_locations, train_index, track_length)

    # apply smooth proportional slowdown
    if distance < slowing_distance_threshold:
        factor = distance / slowing_distance_threshold
        return speed * factor

    return speed


def apply_train_custom_slowdown(speed, train_locations, train_index, track_length, param):
    """
    Custom train slowdown strategy based on distance to next train.
    Operator-defined heuristic: slows down if too close to the train ahead.
    """
    # compute ccw distance to next train
    distance = _distance_to_next_train_ahead(train_locations, train_index, track_length)
    num_trains = len(train_locations)

    # set threshold based on param or heuristic
    threshold = param if param is not None else track_length / (2 * num_trains)

    # reduce speed if train too close
    if distance < threshold:
        return speed * 0.5

    return speed

############################################################
# simulation analysis
############################################################


# 5.1 varying environmental effects
def analyze_and_plot_environment_effects(
    stop_locations,
    speed,
    num_steps,
    track_length,
    num_trials,
    parameter_name,
    track_params_options=None,
    track_slowdown_fn=None,
    train_param=None,
    train_slowdown_fn=None,
):
    """
    Analyze and plot the effect of varying a track parameter on average wait time.
    Parameters:
        stop_locations (list): list of stop locations
        speed (float): standard speed of each train (miles per time step)
        num_steps (int): number of time steps to simulate
        num_trials (int): number of trials to run
        parameter_name (str): name of the track parameter being varied (for labeling purposes)
        track_params_options (list): list of values for the track parameter to test
        track_slowdown_fn (callable): function that applies the track slowdown
        train_param (float): fixed parameter for the train slowdown function
        train_slowdown_fn (callable): function that applies the train slowdown
    """
    # check that track_params_options is provided
    if track_params_options is None:
        raise ValueError("Must provide track_params_options list for varying parameter values.")

    means = []
    stds = []

    # Loop through parameter values and run Monte Carlo
    for param in track_params_options:
        mean_wait, std_wait, _ = run_monte_carlo_realistic(
            stop_locations=stop_locations,
            speed=speed,
            num_steps=num_steps,
            track_length=track_length,
            num_trials=num_trials,
            track_params=[param],
            track_slowdown_fns=[track_slowdown_fn],
            train_params=[train_param] if train_param else None,
            train_slowdown_fns=[train_slowdown_fn] if train_slowdown_fn else None,
        )
        means.append(mean_wait)
        stds.append(std_wait)

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.errorbar(track_params_options, means, yerr=stds, fmt='-o', color='tab:blue', ecolor='red', capsize=4)
    plt.title(f"Effect of {parameter_name} on Average Wait Time")
    plt.xlabel(parameter_name)
    plt.ylabel("Average Wait Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 5.2 varying self-controlled effects
def analyze_and_plot_self_controlled(
    stop_locations,
    speed,
    num_steps,
    track_length,
    num_trials,
    parameter_name,
    track_param=None,
    track_slowdown_fn=None,
    train_param_options=None,
    train_slowdown_fn=None,
):
    """
    Analyze and plot the effect of varying a train parameter on average wait time.
    automatically apply no passing train slowdown function.
    Parameters:
        stop_locations (list): list of stop locations
        speed (float): standard speed of each train (miles per time step)
        num_steps (int): number of time steps to simulate
        num_trials (int): number of trials to run
        parameter_name (str): name of the train parameter being varied (for labeling purposes)
        track_param (float): fixed parameter for the track slowdown function
        track_slowdown_fn (callable): function that applies the track slowdown
        train_param_options (list): list of values for the train parameter to test
        train_slowdown_fn (callable): function that applies the train slowdown
    """
    # check that train_param_options is provided
    # check that train_param_options is provided
    if train_param_options is None:
        raise ValueError("must provide train_param_options list for varying parameter values.")

    means = []
    stds = []

    # loop through parameter values and run monte carlo for each
    for param in train_param_options:
        mean_wait, std_wait, _ = run_monte_carlo_realistic(
            stop_locations=stop_locations,
            speed=speed,
            num_steps=num_steps,
            track_length=track_length,
            num_trials=num_trials,
            track_params=[track_param] if track_param else None,
            track_slowdown_fns=[track_slowdown_fn] if track_slowdown_fn else None,
            train_params=[param],
            train_slowdown_fns=[apply_train_no_pass, train_slowdown_fn],
        )
        means.append(mean_wait)
        stds.append(std_wait)

    # create the plot with error bars
    plt.figure(figsize=(7, 5))
    plt.errorbar(train_param_options, means, yerr=stds, fmt='-o', color='tab:blue', ecolor='red', capsize=4)

    # add horizontal baseline at first mean value
    baseline = means[0]
    plt.axhline(y=baseline, color='gold', linestyle='--', label=f'baseline = {baseline:.2f}')

    # add second labeled line (if desired for clarity)
    plt.axhline(y=means[0], color='gold', linestyle='--', label='baseline (first mean)')

    # finalize the plot
    plt.title(f"effect of {parameter_name} on average wait time")
    plt.xlabel(parameter_name)
    plt.ylabel("average wait time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

############################################################
# independent testing
############################################################

############################################################
# setup
############################################################

random.seed(0)  # maintains consistency across manual tests, do not modify.

# choose a scenario to test and comment out the other

# general stop and train data (not specific to a scenario)
stop_names = [
    "Alewife",
    "Davis",
    "Porter",
    "Harvard",
    "Central",
    "Kendall/MIT",
    "Charles/MGH",
    "Park Street",
]
train_names = ["Andrew", "Beth", "Cody", "Diana", "Eli", "Fiona", "George", "Hannah"]

# # scenario 1 data (evenly spaced stops)
# inter_stop_distance = 2
# locations = [i * inter_stop_distance for i in range(len(stop_names))]
# track_length = inter_stop_distance * len(stop_names)
# speed = 1 # mile per time step

# scenario 2 data
locations = [0, 1.2, 1.9, 2.7, 3.3, 3.9, 4.7, 5.3]
track_length = 7
speed = 0.25  # quarter mile per time step


assert len(locations) == len(
    train_names
), "invalid setup: number of trains must equal number of stops"

NUM_STEPS = 500 # number of time steps for each simulation
NUM_TRIALS = 8 # number of trials used for monte carlo simulation

FRAME_RATE = 4  # adjusts animation speed


############################################################
# ideal example
############################################################

def ideal_example():
    # single run
    location_history = simulate_trains_ideal(locations, speed, NUM_STEPS, track_length)
    avg_wait_time, std_wait_time = get_intertrain_stats(location_history, locations)
    print(f"Average intertrain wait time for single trial: {avg_wait_time}")
    print(f"Standard deviation of intertrain wait time for single trial: {std_wait_time}")

    ani = produce_animation(
        location_history, stop_names, locations, train_names, track_length, show=False
    )
    ani.save("ideal.gif", writer="pillow", fps=FRAME_RATE)

    # # monte carlo simulation
    avg, std, _ = run_monte_carlo_ideal(
        locations, speed, NUM_STEPS, track_length, NUM_TRIALS
    )

    print(f"Average intertrain wait time across {NUM_TRIALS} trials: {avg}")


############################################################
# realistic example
############################################################

def realistic_example():
    p = 0.5
    sigma = 12 / 60

    halting_mean, halting_std, halting_history = run_monte_carlo_realistic(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        track_params=[p],
        track_slowdown_fns=[apply_track_halt],
    )
    halting_ani = produce_animation(
        halting_history, stop_names, locations, train_names, track_length, show=False
    )
    halting_ani.save("halting.gif", writer="pillow", fps=FRAME_RATE)
    print(f"Intertrain wait time stats on halting track: {halting_mean=}, {halting_std=}")

    gaussian_mean, gaussian_std, gaussian_history = run_monte_carlo_realistic(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        track_params=[sigma],
        track_slowdown_fns=[apply_track_gaussian_slow],
    )
    gaussian_ani = produce_animation(
        gaussian_history, stop_names, locations, train_names, track_length, show=False
    )
    gaussian_ani.save("gaussian.gif", writer="pillow", fps=FRAME_RATE)
    print(
        f"Intertrain wait time stats on gaussian track: {gaussian_mean=}, {gaussian_std=}"
    )

    halting_gaussian_mean, halting_gaussian_std, halting_gaussian_history = (
        run_monte_carlo_realistic(
            locations,
            speed,
            NUM_STEPS,
            track_length,
            NUM_TRIALS,
            track_params=[p, sigma],
            track_slowdown_fns=[apply_track_halt, apply_track_gaussian_slow],
        )
    )
    halting_gaussian_ani = produce_animation(
        halting_gaussian_history,
        stop_names,
        locations,
        train_names,
        track_length,
        show=False,
    )
    halting_gaussian_ani.save(
        "halting-and-gaussian.gif", writer="pillow", fps=FRAME_RATE
    )
    print(
        f"Intertrain wait time stats on halting + gaussian track: {halting_gaussian_mean=}, {halting_gaussian_std=}"
    )

    uniform_mean, uniform_std, uniform_history = run_monte_carlo_realistic(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        track_params=[None],
        track_slowdown_fns=[apply_track_uniform_slow],
    )
    uniform_ani = produce_animation(
        uniform_history, stop_names, locations, train_names, track_length, show=False
    )
    uniform_ani.save("uniform.gif", writer="pillow", fps=FRAME_RATE)
    print(f"Intertrain wait time stats on uniform track: {uniform_mean=}, {uniform_std=}")


############################################################
# strategic example
############################################################

def strategic_example():
    halting_p = 0.5

    controlled_mean, controlled_std, controlled_history = run_monte_carlo_realistic(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        track_params=[halting_p],
        track_slowdown_fns=[apply_track_halt],
        train_params=[None],
        train_slowdown_fns=[apply_train_no_pass],
    )
    controlled_ani = produce_animation(
        controlled_history, stop_names, locations, train_names, track_length, show=False
    )
    controlled_ani.save("controlled-nopass.gif", writer="pillow", fps=FRAME_RATE)

    distance_threshold = 0.5
    controlled_smart_mean, controlled_smart_std, controlled_smart_history = (
        run_monte_carlo_realistic(
            locations,
            speed,
            NUM_STEPS,
            track_length,
            NUM_TRIALS,
            track_params=[halting_p],
            track_slowdown_fns=[apply_track_halt],
            train_params=[None, distance_threshold],
            train_slowdown_fns=[apply_train_no_pass, apply_train_smart_slowdown],
        )
    )
    controlled_smart_ani = produce_animation(
        controlled_smart_history,
        stop_names,
        locations,
        train_names,
        track_length,
        show=False,
    )
    controlled_smart_ani.save("controlled-smart.gif", writer="pillow", fps=FRAME_RATE)

    print(f"Intertrain wait time with no train passing: {controlled_mean}")
    print(
        f"Intertrain wait time with controlled slowing and no train passing: {controlled_smart_mean}"
    )


############################################################
# plot and analyze
############################################################

def plot_and_analyze():
    halting_probabilities = [i / 10 for i in range(0, 9)]
    sigmas = [i / 10 for i in range(0, 40)]

    halting_p = 0.5
    gaussian_sigma = 2
    uniform_param = None
    distance_thresholds_in_halting = [i / 50 for i in range(0, 50)]
    distance_thresholds_in_gaussian = [i / 50 for i in range(0, 50)]
    distance_thresholds_in_uniform = [i / 50 for i in range(0, 50)]

    analyze_and_plot_environment_effects(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        parameter_name="Halt Probability",
        track_params_options=halting_probabilities,
        track_slowdown_fn=apply_track_halt,
        train_param=None,
        train_slowdown_fn=apply_train_no_pass,
    )
    analyze_and_plot_environment_effects(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        parameter_name="Sigma",
        track_params_options=sigmas,
        track_slowdown_fn=apply_track_gaussian_slow,
        train_param=None,
        train_slowdown_fn=apply_train_no_pass,
    )
    analyze_and_plot_self_controlled(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        parameter_name="Distance Threshold (on halting track)",
        track_slowdown_fn=apply_track_halt,
        track_param=halting_p,
        train_param_options=distance_thresholds_in_halting,
        train_slowdown_fn=apply_train_smart_slowdown,
    )
    analyze_and_plot_self_controlled(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        parameter_name="Distance Threshold (on gauss track)",
        track_slowdown_fn=apply_track_gaussian_slow,
        track_param=gaussian_sigma,
        train_param_options=distance_thresholds_in_gaussian,
        train_slowdown_fn=apply_train_smart_slowdown,
    )
    analyze_and_plot_self_controlled(
        locations,
        speed,
        NUM_STEPS,
        track_length,
        NUM_TRIALS,
        parameter_name="Distance Threshold (on uniform track)",
        track_slowdown_fn=apply_track_uniform_slow,
        track_param=uniform_param,
        train_param_options=distance_thresholds_in_uniform,
        train_slowdown_fn=apply_train_smart_slowdown,
    )


if __name__ == "__main__":
    # to swap between the two given scenarios (or customize your own),
    # see the section labeled "setup"

    # Uncomment to test!
    #ideal_example()
    #realistic_example()
    #strategic_example()
    plot_and_analyze()

    pass

"""
6.1000 Fall 2025
Problem Set 2

Please fill out the following info:
Name: Samhita Pokkunuri
Kerberos: samhitap
Approximate time spent (HH:MM): 5 hours, 0 min
"""

import matplotlib.pyplot as plt


############################################################
# reading and cleaning data
############################################################


def clean_data(x_vals, y_vals):
    """
    Discard malformed entries from a dataset.

    Parameters:
        x_vals (list): Contains strs representing the x-values of a dataset.
        y_vals (list): Contains strs representing the y-values of a dataset.

    Any entry in x_vals that does not represent a valid number should be
    discarded along with the corresponding entry in y_vals, and vice versa.
    These values should be deleted from the input lists directly.
    """
    cleanedX = []
    cleanedY = []
    for i in range(len(min(x_vals,y_vals))):
        try: # check if elements are floats or not
            float(x_vals[i])
            float(y_vals[i])
            cleanedX.append(x_vals[i])
            cleanedY.append(y_vals[i])
        except ValueError:
            continue
    x_vals[:] = cleanedX # cleaned x list
    y_vals[:] = cleanedY # cleaned y list
    return x_vals,y_vals
    raise NotImplementedError


def read_data(filename):
    """
    Convert the data in a specified file into lists of x- and y-values.

    Parameters:
        filename (str): The name of the file to be read.

    Each line in the file should be of the form
        x_val,y_val
    where x_val and y_val are decimal numbers.

    Return a list [x_vals, y_vals], where x_vals and y_vals are lists of
    floats, having the same length.
    """

    dust = []
    efficiency = []

    with open(filename, 'r') as file:
        for line in file:
            x_val, y_val = line.strip().split(",") # getting x and y values
            dust.append(x_val)
            efficiency.append(y_val)

    clean_data(dust, efficiency) # cleaning data for non-floats

    dust_float = []
    for x in dust:
        dust_float.append(float(x)) # re-convert to floats

    efficiency_float = []
    for y in efficiency:
        efficiency_float.append(float(y)) # re-convert to floats

    return [dust_float, efficiency_float]



############################################################
# mean squared error
############################################################


def mean_squared_error(y_actual, y_pred):
    """
    Calculate the mean squared error (MSE) between actual and predicted y-values.

    Parameters:
        y_actual (list): Contains floats representing y-values from data.
        y_pred (list): Contains floats representing y-values predicted
            from applying a model to the data's x-values, in the same
            order as data points in y_actual.

    Return the calculated MSE as a float.
    """
    mse = 0.0
    n = len(y_actual)
    for i in range(0,n):
        mse += (y_actual[i] - y_pred[i]) ** 2 # continually add the mse value for each y value
    return mse / n



############################################################
# fitting a linear model
############################################################


def predict_y(x_val, coefficients):
    """
    Evaluate a polynomial model on a given x-value to predict its
    associated y-value.

    Parameters:
        x_val (float): An x-value on which to evaluate the model.
        coefficients (list): Specifies the coefficients of a polynomial
            in order of decreasing degree. The last coefficient is the
            constant term.

    Return the predicted y-value as a float.
    """
    y_val = 0
    for i in range(0,len(coefficients)):
        y_val += coefficients[i] * (x_val ** (len(coefficients)-1-i)) # polynominal model
    return y_val
    raise NotImplementedError


def fit_linear_model(x_vals, y_vals, a_range, b_range, precision):
    """
    Find a linear model that minimizes the MSE on the provided datapoints.

    Parameters:
        x_vals (list): A list of floats representing x-values.
        y_vals (list): A list of floats representing y-values.
        a_range (list): A list of two floats indicating the lower and
            upper bounds on the "a" coefficient.
        b_range (list): A list of two floats indicating the lower and
            upper bounds on the "b" coefficient.
        precision (float): How close to the true optimal values of "a"
            and "b" our answer needs to be.

    Return a list [a, b] of float values representing the model y = ax + b
    such that if a* and b* are the true values that minimize the MSE, then
    a and b are within precision of a* and b*, respectively.
    """
    error = float("inf")
    aValues = []
    bValues = []
    values = []

    a = a_range[0]
    while a <= a_range[1]: # with precision differences, go from min a to max a
        aValues.append(a)
        a += precision

    b = b_range[0] # with precision differences, go from min b to max b
    while b <= b_range[1]:
        bValues.append(b)
        b += precision

    for i in range(len(aValues)):
        for j in range(len(bValues)):
            # build predictions for this (a, b)
            y_pred = []
            for value in range(len(x_vals)):
                theoreticalY = aValues[i] * x_vals[value] + bValues[j]
                y_pred.append(theoreticalY)

            # compute MSE
            counterError = mean_squared_error(y_vals, y_pred)

            if counterError < error:
                error = counterError
                values = [aValues[i], bValues[j]]




    return values
    raise NotImplementedError


############################################################
# running regression on linear model data and plotting
############################################################


def solar_dust_regression(filename):
    """
    Read the data on solar panel efficiency and dust levels, fit a
    linear model to it, plot both the data and model, and indicate the
    dust amount corresponding to when cleaning is needed. See the pset
    webpage for plotting specifications.
    """
    x_vals, y_vals = read_data("solar_panel_data.txt")

    a_range = [-1.0, 0.0]
    precision = 0.03

    # using exhaustive search
    #b_range = [50.0, 105.0]
    #a, b = fit_linear_model(x_vals, y_vals, a_range, b_range, precision)

    # using bisection search
    a, b = fit_linear_model_bisection(x_vals, y_vals, a_range, precision)

    x_curve = []
    y_curve = []
    x_point = min(x_vals)
    while x_point <= max(x_vals):
        x_curve.append(x_point)
        y_curve.append(predict_y(x_point, [a, b]))
        x_point += 0.1

    threshold_y = 75
    threshold_x = (threshold_y - b) / a

    # plotting scatterpoints and line
    plt.scatter(x_vals, y_vals, color='blue', label='Solar panel data')
    plt.plot(x_curve, y_curve, color='red', label='Linear fit')

    # threshold
    plt.axhline(y=threshold_y, color='green', linestyle='--')
    plt.axvline(x=threshold_x, color='green', linestyle='--')

    # threshold text
    plt.text(threshold_x, threshold_y - 5, "x = " + str(round(threshold_x, 2)) + " g", color='green')
    plt.text(min(x_vals), threshold_y + 1, "y = " + str(threshold_y) + " %", color='green')

    # labels
    plt.title("Dust Mass Threshold for Cleaning Solar Panels")
    plt.xlabel("Dust mass (g)")
    plt.ylabel("Solar Panel Efficiency (%)")
    plt.legend()
    plt.show()


    # raise NotImplementedError


############################################################
# fitting a quadratic model
############################################################


def fit_quadratic_model(x_vals, y_vals, a_range, b_range, c_range, precision):
    """
    Objective: Find coefficiencts a, b, c for the quadtratic model to minimize the MSE using exhaustive enumeration.

    Parameters:
        x_vals (list): A list of floats representing x-values.
        y_vals (list): A list of floats representing y-values.
        a_range (list): A list of two floats indicating the lower and
            upper bounds on the "a" coefficient.
        b_range (list): A list of two floats indicating the lower and
            upper bounds on the "b" coefficient.
        c_range (list): A list of two floats indicating the lower and
            upper bounds on the "c" coefficient.
        precision (float): How close to the true optimal values of "a", "b",
            and "c" our answer needs to be.

    The function should return coefficients a, b, and c that minimize the MSE.
    """

    error = float("inf")
    aValues = []
    bValues = []
    cValues = []
    values = []

    a = a_range[0] # with precision differences, go from min a to max a
    while a <= a_range[1]:
        aValues.append(a)
        a += precision

    b = b_range[0] # with precision differences, go from min b to max b
    while b <= b_range[1]:
        bValues.append(b)
        b += precision

    c = c_range[0] # with precision differences, go from min c to max c
    while c <= c_range[1]:
        cValues.append(c)
        c += precision

    for i in range(len(aValues)):
        for j in range(len(bValues)):
            for k in range(len(cValues)):
                # build predictions for this (a, b, c)
                y_pred = []
                for value in range(len(x_vals)):
                    theoreticalY = predict_y(x_vals[value],[aValues[i],bValues[j],cValues[k]])
                    y_pred.append(theoreticalY)

                # compute MSE
                counterError = mean_squared_error(y_vals, y_pred)

                if counterError < error:
                    error = counterError
                    values = [aValues[i], bValues[j],cValues[k]]




    return values

    raise NotImplementedError


############################################################
# running regression on quadratic model data and plotting
############################################################


def sunlight_regression():
    """
    Read the data on sunlight exposure and plan heights, fit a quadratic
    model to it, plot both the data and model, and indicate the
    amount of sunlight exposure for optimal plant growth. See the pset
    webpage for plotting specifications.
    """
    sunlight, growth = read_data("plant_data.txt")

    x_vals, y_vals = read_data("plant_data.txt")
    a_range = [-7.0, -5.0]
    b_range = [60.0, 70.0]
    c_range = [-35.0, -30.0]
    precision = 0.03
    coefficients = fit_quadratic_model(x_vals, y_vals, a_range, b_range, c_range, precision)

    # creating parabola
    x_curve = []
    y_curve = []
    step = 0.1
    x_point = min(x_vals)
    while x_point <= max(x_vals):
        x_curve.append(x_point)
        y_curve.append(predict_y(x_point, coefficients))
        x_point += step

    # max and optimal points of curve
    max_y = max(y_curve)
    optimal_x = x_curve[y_curve.index(max_y)]

    # scatterplot, actual plot, and then labels
    plt.scatter(x_vals, y_vals, color='blue', label='Plant growth data')
    plt.plot(x_curve, y_curve, color='red', label='Quadratic fit')

    # most optimal point recognized
    plt.axvline(x=optimal_x, color='green', linestyle='--')
    plt.axhline(y=max_y, color='green', linestyle='--')

    # labels (including labeling most optimal point)
    plt.text(optimal_x + 0.2, 0, f"x = {optimal_x:.2f}", color='green')
    plt.text(0, max_y + 0.2, f"y = {max_y:.2f}", color='green')
    plt.title("Plant Growth vs Sunlight Exposure")
    plt.xlabel("Sunlight exposure (hours per day)")
    plt.ylabel("Plant growth (mm/day)")
    plt.legend()
    plt.show()
    raise NotImplementedError


############################################################
# fitting a linear model using bisection search
############################################################


def slice_mse_model(x_vals, y_vals, a):
    """
    Given data on which to fit a linear model y = a*x+ b, determine the
    MSE quadratic polynomial over the parameter "b" that expresses the
    MSE when the parameter "a" is fixed at a given value.

    Parameters:
        x_vals (list): The same as in find_min_on_slice().
        y_vals (list): The same as in find_min_on_slice().
        a (float): The same as in find_min_on_slice().

    Return a list of float coefficients [alpha, beta, gamma] specifying
    the desired quadratic polynomial (alpha * b**2 + beta * b + gamma)
    in terms of "b".
    """

    assert len(x_vals) == len(y_vals)
    num_points = len(x_vals)
    alpha = 0
    beta = 0
    gamma = 0
    for x, y in zip(x_vals, y_vals):
        alpha += 1
        beta += 2*a*x - 2*y
        gamma += y**2 + (a*x)**2 - 2*y*a*x
    alpha /= num_points
    beta /= num_points
    gamma /= num_points
    return [alpha, beta, gamma]


def find_min_on_slice(x_vals, y_vals, a):
    """
    Given data on which to fit a linear model y = a*x+ b, find the
    minimum value of the MSE when the parameter "a" is fixed at a given
    value.
    Finding minimum a value, using that to calculate a b value.

    Parameters:
        x_vals (list): Contains floats representing x-values of a dataset.
        y_vals (list): Contains floats representing y-values of a dataset.
        a (float): The fixed value of the parameter "a" for a linear model.

    Return a float representing the minimum MSE value along the "a" slice.
    """
    y_pred = []
    alpha, beta, gamma = slice_mse_model(x_vals, y_vals, a) # using helper

    b_opt = -beta / (2 * alpha) # using formula to determine b value

    for x in x_vals:
        y_pred.append(a*x + b_opt)

    return mean_squared_error(y_vals, y_pred)

    raise NotImplementedError

def fit_linear_model_bisection(x_vals, y_vals, a_range, precision):
    """
    Find a linear model, similarly to fit_linear_model(), but using
    bisection search on possible values of the "a" parameter.

    Parameters:
        x_vals (list): The same as in fit_linear_model().
        y_vals (list): The same as in fit_linear_model().
        a_range(list): The same as in fit_linear_model().
        precision (float): The same as in fit_linear_model(), except it
            applies only to the "a" parameter.

    Return a list [a, b] of float values representing the model y = ax + b,
    the same as in fit_linear_model(), but without the requirement on
    the precision of "b".
    """
    a_min, a_max = a_range
    best_a = None
    best_b = None
    min_mse = float("inf")

    while a_max - a_min > precision:
        a_mid = (a_min + a_max) / 2

        # evaluate MSE
        mse_left = find_min_on_slice(x_vals, y_vals, a_mid - precision/10)
        mse_mid  = find_min_on_slice(x_vals, y_vals, a_mid)
        mse_right= find_min_on_slice(x_vals, y_vals, a_mid + precision/10)

        # narrow the search
        if mse_left < mse_mid:
            a_max = a_mid
        elif mse_right < mse_mid:
            a_min = a_mid
        else:
            a_min = a_mid - precision/10
            a_max = a_mid + precision/10

        # update best values
        if mse_mid < min_mse:
            min_mse = mse_mid
            # use slice_mse_model to find optimal b for current best a
            alpha, beta, gamma = slice_mse_model(x_vals, y_vals, a_mid)
            best_b = -beta / (2 * alpha)
            best_a = a_mid

    return [best_a, best_b]

    raise NotImplementedError


############################################################
# manual tests
############################################################


if __name__ == "__main__":
    pass
    solar_dust_regression("solar_panel_data.txt")
    #sunlight_regression()

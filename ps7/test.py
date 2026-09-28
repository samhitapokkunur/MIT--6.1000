import contextlib
import io
import os
import random
import unittest
from scipy.stats import linregress
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import pandas as pd
from helpers import Results_600, TestProblemSetBase, case_options, testsuite_options, run_student_script

# Path relative to THIS test file
BASE_DIR = os.path.dirname(__file__)

# Path to the real data/ directory that ships with the pset
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_FILE = os.path.join(DATA_DIR, "temp_change.csv")
DISASTERS_FILE = os.path.join(DATA_DIR, "disasters.csv")
POP_FILE = os.path.join(DATA_DIR, "population.json")

@testsuite_options(10, 4)
class TestPart2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}

    @case_options(
        1,
        failure="Function 'load_data' is not implemented correctly",
        error="Error occurred while testing 'load_data'",
    )
    def test_load_data(self):
        """Test that load_data returns a DataFrame with expected columns."""
        # os.makedirs("test", exist_ok=True)
        if not os.path.exists("data/sample_temperature_data.csv"):
            df_test = pd.DataFrame(
                {"Year": [2000, 2001, 2002], "Temperature": [14.1, 14.3, 14.5]}
            )
            df_test.to_csv("data/sample_temperature_data.csv", index=False)

        if not os.path.exists("data/sample_population_data.json"):
            df_test = pd.DataFrame(
                [
                    {
                        "Entity": "A",
                        "Code": "AAA",
                        "Year": 2000,
                        "Population (historical)": 1000,
                    },
                    {
                        "Entity": "A",
                        "Code": "AAA",
                        "Year": 2001,
                        "Population (historical)": 1100,
                    },
                    {
                        "Entity": "A",
                        "Code": "AAA",
                        "Year": 2002,
                        "Population (historical)": 1200,
                    },
                ]
            )
            df_test.to_json("data/sample_population_data.json",
                            orient="records")

        fn = self.global_vars.get("load_data")
        if not callable(fn):
            self.fail(
                "Function 'load_data' is not defined or not callable in the student script."
            )

        df = fn("data/sample_temperature_data.csv")

        self.assertIsInstance(
            df, pd.DataFrame, "Function 'load_data' must return a DataFrame"
        )

        expected_columns = {"Year", "Temperature"}

        self.assertTrue(
            expected_columns.issubset(set(df.columns)),  # type: ignore
            # type: ignore
            f"DataFrame columns {df.columns} do not include expected columns {expected_columns}.",
        )

        df = fn("data/sample_population_data.json")

        expected_columns = {"Entity", "Code",
                            "Year", "Population (historical)"}
        self.assertTrue(
            expected_columns.issubset(set(df.columns)),
            f"DataFrame columns {df.columns} do not include expected columns {expected_columns}",
        )

    @case_options(
        1,
        failure="Function 'process_temperature_data' did not correctly clean or rename temperature data",
        error="Error occurred while testing 'process_temperature_data'",
    )
    def test_process_temp_change(self):
        """Test process_temperature_data function with sample data"""
        raw = pd.DataFrame(
            {
                "ObjectId": [1, 2, 3],
                "Country": ["A", "B", "C"],
                "ISO2": ["AA", "BB", "CC"],
                "ISO3": ["AAA", "BBB", "CCC"],
                "Indicator": [
                    "Temperature change from baseline (°C)",
                    "Temperature change from baseline (°C)",
                    "Temperature change from baseline (°C)",
                ],
                "Unit": ["Celsius", "Celsius", "Celsius"],
                "Source": ["X", "Y", "Z"],
                "CTS Code": [101, 102, 103],
                "CTS Name": ["Temp", "Temp", "Temp"],
                "CTS Full Descriptor": ["Temp Desc", "Temp Desc", "Temp Desc"],
                "2000": [0.5, 0.6, 0.7],
                "2001": [0.55, 0.65, 0.75],
                "2002": [0.6, 0.7, 0.8],
            }
        )

        fn = self.global_vars.get("process_temperature_data")
        if not callable(fn):
            self.fail(
                "Function 'process_temperature_data' is not defined or not callable in the student script."
            )

        df: pd.DataFrame = fn(raw)  # type: ignore

        self.assertIsInstance(
            df,
            pd.DataFrame,
            "Function 'process_temperature_data' must return a DataFrame",
        )

    @case_options(
        1,
        failure="Function 'process_disasters' failed to correctly aggregate disaster data",
        error="Error occurred while testing 'process_disasters'",
    )
    def test_process_disasters(self):
        """Test process_disaster_data function with sample data"""
        raw = pd.DataFrame(
            {
                "ObjectId": [1, 2, 3],
                "Country": ["A", "B", "C"],
                "ISO2": ["AA", "BB", "CC"],
                "ISO3": ["AAA", "BBB", "CCC"],
                "Indicator": [
                    "Number of disasters",
                    "Number of disasters",
                    "Number of disasters",
                ],
                "Unit": ["Count", "Count", "Count"],
                "Source": ["X", "Y", "Z"],
                "CTS Code": [101, 102, 103],
                "CTS Name": ["Disaster", "Disaster", "Disaster"],
                "CTS Full Descriptor": [
                    "Disaster Desc",
                    "Disaster Desc",
                    "Disaster Desc",
                ],
            }
        )

        fn = self.global_vars.get("process_disaster_data")
        if not callable(fn):
            self.fail(
                "Function 'process_disaster_data' is not defined or not callable in the student script."
            )

        df_disasters: pd.DataFrame = fn(raw, people=False)  # type: ignore
        df_people: pd.DataFrame = fn(raw, people=True)  # type: ignore

        self.assertIsInstance(
            df_disasters,
            pd.DataFrame,
            "Function 'process_disaster_data' must return a DataFrame when people=False",
        )
        self.assertIsInstance(
            df_people,
            pd.DataFrame,
            "Function 'process_disaster_data' must return a DataFrame when people=True",
        )

    @case_options(
        1,
        failure="Function 'process_population_data' did not correctly format population data",
        error="Error occurred while testing 'process_population_data'",
    )
    def test_process_population_data(self):
        """Test process_population_data function with sample data."""
        raw = pd.DataFrame(
            {
                "Entity": ["A", "A", "B", "B"],
                "Code": ["AAA", "AAA", "BBB", "BBB"],
                "Year": [2000, 2001, 2000, 2001],
                "Population (historical)": ["1000", "1100", "2000", "2100"],
            }
        )

        fn = self.global_vars.get("process_population_data")
        if not callable(fn):
            self.fail(
                "Function 'process_population_data' is not defined or not callable in the student script"
            )

        df: pd.DataFrame = fn(raw)  # type: ignore

        self.assertIsInstance(
            df,
            pd.DataFrame,
            "Function 'process_population_data' must return a DataFrame",
        )

    @case_options(
        1,
        failure="Function 'country_to_continent' did not correctly map ISO codes to continent names",
        error="Error occurred while testing 'country_to_continent'",
    )
    def test_country_to_continent(self):
        """Test country_to_continent function with various ISO3 codes"""
        fn = self.global_vars.get("country_to_continent")
        if not callable(fn):
            self.fail(
                "Function 'country_to_continent' is not defined or not callable in the student script."
            )
        result1 = fn("USA")
        result2 = fn("FRA")
        result3 = fn("XXX")  # invalid ISO
        result4 = fn("AZO")  # special case
        result5 = fn("ANT")  # special case
        self.assertIn(result1, ["North America"],
                      "Incorrect continent for USA")
        self.assertIn(result2, ["Europe"], "Incorrect continent for FRA")
        self.assertTrue(
            result3 is None,
            "Function should handle invalid ISO codes gracefully (e.g., return None)",
        )
        self.assertIn(
            result4,
            ["Europe"],
            "Incorrect continent for AZO. Be sure to handle all special cases",
        )
        self.assertIn(
            result5,
            ["North America"],
            "Incorrect continent for ANT. Be sure to handle all special cases",
        )

@testsuite_options(10, 4)
class TestPart3(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}
    # 3.1 global mean temperature vs. year
    @case_options(
        1,
        failure="Function 'mean_temp_over_time' did not correctly plot global mean temperature vs. time",
        error="Error occurred while testing 'mean_temp_over_time'",
    )
    def test_mean_temp_over_time(self):
        """Test mean_temp_over_time function by checking axis values and plot features"""
        fn = self.global_vars.get("mean_temp_over_time")
        if not callable(fn):
            self.fail(
                "Function 'mean_temp_over_time' is not defined or not callable in the student script."
            )
        # additions
        student_load_data = self.global_vars.get("load_data")
        student_proc_temp = self.global_vars.get("process_temperature_data")
        df_temp_unproc = student_load_data(TEMP_FILE)
        df_temp = student_proc_temp(df_temp_unproc)

        fig1, ax1 = fn(df_temp, 1, show_plot=False)
        x1 = ax1.lines[0].get_xdata()
        y1 = ax1.lines[0].get_ydata()

        fig2, ax2 = fn(df_temp, 5, show_plot=False)
        x2 = ax2.lines[0].get_xdata()
        y2 = ax2.lines[0].get_ydata()

        self.assertEqual(
            len(x1), 64, "incorrect number of years as x-values"
        )  # check range of years used to plot
        self.assertTrue(
            all(x1[i] <= x1[i + 1] for i in range(len(x1) - 1))
        )  # check that x vals are increasing order

        slope, *_ = linregress(range(len(y1)), y1)
        self.assertGreater(
            slope,
            0,
            "slope is negative or has nans. make sure to specify min_periods=1 with rolling average",
        )  # check for positive slope

        # check that moving average lowers variation
        dy_raw = np.diff(y1)
        dy_smooth = np.diff(y2)
        std_raw = np.nanstd(dy_raw)
        std_smooth = np.nanstd(dy_smooth)

        self.assertLess(
            std_smooth,
            std_raw,
            "Moving average should produce a smoother (less variable) curve. Calculated variations don't reflect this.",
        )

        # check x- and y-labels, titles
        xlabel1 = ax1.get_xlabel().lower()
        xlabel2 = ax2.get_xlabel().lower()
        ylabel1 = ax1.get_ylabel().lower()
        ylabel2 = ax2.get_ylabel().lower()
        title1 = ax1.get_title()
        title2 = ax2.get_title()

        self.assertEqual("year", xlabel1.lower(), "x-axis must contain the word 'Year' for all plots in this section.")
        self.assertEqual("year", xlabel2.lower(), "x-axis must contain the word 'Year' for all plots in this section.")
        self.assertEqual("mean temperature change (°c)", ylabel1.lower(),
                      "y-axis must mention 'temperature' for all plots in this section.")
        self.assertEqual("mean temperature change (°c)", ylabel2.lower(),
                      "y-axis must mention 'temperature' for all plots in this section.")
        self.assertTrue(
            len(title1) > 0,
            "The plot must have a title."
        )
        self.assertTrue(
            len(title2) > 0,
            "The plot must have a title."
        )

    # 3.2 temperatures and disasters
    @case_options(
        1,
        failure="Function 'temps_and_disasters' did not correctly plot temperature and disaster data vs. time",
        error="Error occurred while testing 'temps_and_disasters'",
    )
    def test_temps_and_disasters(self):
        """Test temps_and_disasters function by checking axis values and plot features"""
        fn = self.global_vars.get("temps_and_disasters")
        if not callable(fn):
            self.fail(
                "Function 'temps_and_disasters' is not defined or not callable in the student script."
            )

        student_load_data = self.global_vars.get("load_data")
        student_proc_temp = self.global_vars.get("process_temperature_data")
        student_proc_dis = self.global_vars.get("process_disaster_data")
        df_temp_unproc = student_load_data(TEMP_FILE)
        df_dis_unproc = student_load_data(DISASTERS_FILE)
        df_temp = student_proc_temp(df_temp_unproc)
        df_dis = student_proc_dis(df_dis_unproc)

        fig, ax1, ax2 = fn(df_temp, df_dis, show_plot=False)

        temp_x = ax1.lines[0].get_xdata()
        dis_x = ax2.lines[0].get_xdata()
        y = ax1.lines[0].get_ydata()
        self.assertEqual(
            len(temp_x), 45, "incorrect number of years as x-values")
        self.assertTrue(
            all(temp_x[i] <= temp_x[i + 1] for i in range(len(temp_x) - 1))
        )  # check that x vals are increasing order

        # test that both temp and disaster data generally increasing
        temp_y = ax1.lines[0].get_ydata()
        slope, *_ = linregress(temp_x, temp_y)

        self.assertGreater(
            slope, 0, "Temperature anomaly should generally increase over time."
        )
        dis_y = ax2.lines[0].get_ydata()
        slope_d, *_ = linregress(dis_x, dis_y)

        self.assertGreater(
            slope_d, 0, "Mean disaster count should show a positive trend."
        )
        # check formatting
        xlabel = ax1.get_xlabel().lower()
        self.assertIn("year", xlabel,
            "x-axis label (year) must contain the word 'Year'.")
        ylabel1 = ax1.get_ylabel().lower()
        self.assertIn("temperature", ylabel1,
            "y-axis label (temperature) must contain the word 'temperature'.")
        ylabel2 = ax2.get_ylabel().lower()
        self.assertIn("disaster", ylabel2,
            "y-axis label (disasters) must contain the word 'disaster'.")
        title = ax1.get_title().lower()
        self.assertTrue(
            isinstance(title, str) and len(
                title.strip()) > 0, "Plot must have a title."
        )

@testsuite_options(10, 4)
class TestPart4(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}
    # could also do add_continent test cases but seems simple enough tbh
    # 4.1 skeptical friend, temperature_by_continent
    @case_options(
        1,
        failure="Function 'temperature_by_continent' did not correctly plot temperature vs. time by continent",
        error="Error occurred while testing 'temperature_by_continent'",
    )
    def test_temperature_by_continent(self):
        """Test temperature_by_continent function by checking axis values and plot features"""
        fn = self.global_vars.get("temperature_by_continent")
        if not callable(fn):
            self.fail(
                "Function 'temperature_by_continent' is not defined or not callable in the student script."
            )

        # load and process student fns
        student_load_data = self.global_vars.get("load_data")
        student_proc_temp = self.global_vars.get("process_temperature_data")
        student_add_cont = self.global_vars.get("add_continents")
        # load temp dataset
        df_temp_unproc = student_load_data(TEMP_FILE)
        df_temp = student_proc_temp(df_temp_unproc)
        df_temp = student_add_cont(df_temp)

        fig, ax = fn(df_temp, show_plot=False)

        # check that there are 6 lines
        lines = ax.get_lines()
        self.assertEqual(
            len(lines),
            6,
            "Expected 6 continent lines (Africa, Asia, Europe, North America, Oceania, South America)."
        )
        # check that each continent has many pts
        for line in lines:
            x_vals = line.get_xdata()
            y_vals = line.get_ydata()

            # non-empty line
            self.assertGreater(
                len(x_vals),
                40,  # 60 years typical for dataset
                "Each continent line should have many data points (40+)."
            )
            self.assertEqual(
                len(x_vals), len(y_vals),
                "x and y data lengths must match."
            )

            # check that x vals/years sorted
            self.assertTrue(
                all(x_vals[i] <= x_vals[i+1] for i in range(len(x_vals)-1)),
                "Years on the x-axis must be in increasing order."
            )

        xlabel = ax.get_xlabel().lower()
        ylabel = ax.get_ylabel().lower()
        title = ax.get_title().lower()

        self.assertEqual("year", xlabel.lower(), "x-axis must contain the word 'Year'.")
        self.assertEqual("mean temperature change (°c)", ylabel.lower(),
                      "y-axis must mention 'temperature'.")
        self.assertTrue(
            len(title) > 0,
            "The plot must have a title."
        )

        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")

        expected_conts = {"africa", "asia", "europe",
                          "north america", "oceania", "south america"}
        plotted_conts = {t.get_text().lower() for t in legend.get_texts()}

        self.assertSetEqual(
            expected_conts,
            plotted_conts,
            f"Legend must include exactly these continents: {expected_conts}"
        )

    # 4.2 skeptical friend, per capita people affected analysis
    @case_options(
        1,
        failure="Function 'analyze_disaster_per_capita' did not correctly plot disaster effect per capita vs. time by continent",
        error="Error occurred while testing 'analyze_disaster_per_capita'",
    )
    def test_analyze_disaster_per_capita(self):
        """Test analyze_disaster_per_capita function by checking axis values and plot features"""
        fn = self.global_vars.get("analyze_disaster_per_capita")
        if not callable(fn):
            self.fail(
                "Function 'analyze_disaster_per_capita' is not defined or not callable in the student script."
            )

        # load and process student fns
        student_load_data = self.global_vars.get("load_data")
        student_proc_dis = self.global_vars.get("process_disaster_data")
        student_proc_pop = self.global_vars.get("process_population_data")
        student_add_cont = self.global_vars.get("add_continents")
        # load temp and population dataset
        df_dis_unproc = student_load_data(DISASTERS_FILE)
        df_pop_unproc = student_load_data(POP_FILE)
        df_dis = student_proc_dis(df_dis_unproc, people=True)
        df_pop = student_proc_pop(df_pop_unproc)

        df_dis = student_add_cont(df_dis)

        merged, fig, ax = fn(df_dis, df_pop, show_plot=False)

        # check that there are 6 lines
        lines = ax.get_lines()
        self.assertEqual(
            len(lines),
            6,
            "Expected 6 continent lines (Africa, Asia, Europe, North America, Oceania, South America)."
        )
        # check that each continent has many pts
        for line in lines:
            x_vals = line.get_xdata()
            y_vals = line.get_ydata()

            # non-empty line
            self.assertGreater(
                len(x_vals),
                40,  # 60 years typical for dataset
                "Each continent line should have many data points (40+)."
            )
            self.assertEqual(
                len(x_vals), len(y_vals),
                "x and y data lengths must match."
            )

            # check that x vals/years sorted
            self.assertTrue(
                all(x_vals[i] <= x_vals[i+1] for i in range(len(x_vals)-1)),
                "Years on the x-axis must be in increasing order."
            )

        xlabel = ax.get_xlabel().lower()
        ylabel = ax.get_ylabel().lower()
        title = ax.get_title().lower()

        self.assertIn("year", xlabel, "x-axis must contain the word 'Year'.")
        self.assertIn("people affected", ylabel,
                      "y-axis must mention 'people affected'.")
        self.assertTrue(
            len(title) > 0,
            "The plot must have a title."
        )

        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")

        expected_conts = {"africa", "asia", "europe",
                          "north america", "oceania", "south america"}
        plotted_conts = {t.get_text().lower() for t in legend.get_texts()}

        self.assertSetEqual(
            expected_conts,
            plotted_conts,
            f"Legend must include exactly these continents: {expected_conts}"
        )

@testsuite_options(10, 4)
class TestPart5(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}
    # 5 regression analysis
    @case_options(
        1,
        failure="Function 'run_linear_regression' is not implemented correctly",
        error="Error occurred while testing 'run_linear_regression'",
    )
    def test_run_linear_regression_perfect_positive_slope(self):
        """
        Simple case: perfect line with positive slope.
        y = 2x for x in {0, 1, 2, 3}
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        df = pd.DataFrame(
            {
                "x": [0, 1, 2, 3],
                "y": [0, 2, 4, 6],
            }
        )

        result = fn(df, "x", "y", show_plot=False)

        if isinstance(result, tuple):
            slope, intercept, r2, x_vals, y_pred, fig, ax = result[0], result[
                1], result[2], result[3], result[4], result[5], result[6]
        else:
            self.fail(
                "run_linear_regression should return a tuple containing at least (slope, intercept, r2)."
            )

        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(r2, float)

        self.assertTrue(
            np.isclose(r2, 1.0, atol=0.01),
            "R^2 should be 1.0 (within tolerance) for a perfect linear relationship.",
        )

        expected_slope = 2.0
        expected_intercept = 0.0
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))

    @case_options(
        1,
        failure="Function 'run_linear_regression' fails on noisy positive slopes",
        error="Error occurred while testing 'run_linear_regression' on noisy positive data",
    )
    def test_run_linear_regression_noisy_positive1(self):
        """
        Noisy positive slope data.
        Uses linregress + r2_score as the reference solution.
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        # --- define input data ---
        df = pd.DataFrame(
            {
                "x": np.array([0, 1, 2, 3, 4, 5]),
                "y": np.array([0.1, 1.9, 4.2, 5.8, 8.1, 10.2]),  # roughly y ~ 2x
            }
        )

        result = fn(df, "x", "y", show_plot=False)
        if isinstance(result, tuple) and len(result) >= 7:
            slope, intercept, r2, x_vals, y_pred, fig, ax = result
        else:
            self.fail(
                "run_linear_regression should return at least (slope, intercept, r2, x_vals, y_pred, fig, ax)."
            )

        lr = linregress(df["x"], df["y"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["x"]
        expected_r2 = r2_score(df["y"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should be close to linregress result ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should be close to linregress result ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should be close to r2_score result ({expected_r2}). Got {r2}."
        )

        self.assertIsInstance(
            ax, Axes, "run_linear_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()



        self.assertIsNotNone(legend, "Plot must contain a legend.")

        self.assertEqual(
            "x",
            xlabel.lower(),
            "x-axis label for run_linear_regression should mention 'x'.",
        )
        self.assertEqual(
            "y",
            ylabel.lower(),
            "y-axis label for run_linear_regression should mention 'y'.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "run_linear_regression plots should have a non-empty title.",
        )

    @case_options(
        1,
        failure="Function 'run_linear_regression' fails on negative slope noisy data",
        error="Error occurred while testing 'run_linear_regression' on noisy negative data",
    )
    def test_run_linear_regression_noisy_negative1(self):
        """
        Noisy negative slope data.
        Uses linregress + r2_score as the reference solution.
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        df = pd.DataFrame(
            {
                "x": np.array([0, 1, 2, 3, 4, 5]),
                "y": np.array([10.0, 8.2, 6.1, 4.0, 2.2, 0.1]),  # roughly 10 - 2x
            }
        )

        result = fn(df, "x", "y", show_plot=False)
        if isinstance(result, tuple) and len(result) >= 7:
            slope, intercept, r2, x_vals, y_pred, fig, ax = result
        else:
            self.fail(
                "run_linear_regression should return at least (slope, intercept, r2, x_vals, y_pred, fig, ax)."
            )

        lr = linregress(df["x"], df["y"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["x"]
        expected_r2 = r2_score(df["y"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should be close to linregress result ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should be close to linregress result ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should be close to r2_score result ({expected_r2}). Got {r2}."
        )

        # --- Axes / labeling checks ---
        self.assertIsInstance(
            ax, Axes, "run_linear_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertEqual(
            "x",
            xlabel.lower(),
            "x-axis label for run_linear_regression should mention 'x'.",
        )
        self.assertEqual(
            "y",
            ylabel.lower(),
            "y-axis label for run_linear_regression should mention 'y'.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "run_linear_regression plots should have a non-empty title.",
        )

    @case_options(
        1,
        failure=(
            "Function 'run_linear_regression' fails on offset-domain noisy data "
            "compared to SciPy linregress + r2_score"
        ),
        error="Error occurred while testing 'run_linear_regression' on offset-domain data",
    )
    def test_run_linear_regression_offset_domain_complex(self):
        """More complex case with x-values far from 0 and non-zero intercept.

        This ensures students are not relying on x starting at 0 and that the
        regression matches SciPy's linregress and sklearn's r2_score.
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        # x-values are shifted so they are not small integers starting from 0
        x_vals_input = np.array([1990, 1995, 2000, 2005, 2010, 2015])
        # Underlying relationship: y ≈ 0.5 * x - 950 with a bit of noise
        y_vals_input = 0.5 * x_vals_input - 950 + np.array([0.1, -0.2, 0.05, -0.1, 0.2, -0.15])

        df = pd.DataFrame({"x": x_vals_input, "y": y_vals_input})

        result = fn(df, "x", "y", show_plot=False)
        if isinstance(result, tuple) and len(result) >= 7:
            slope, intercept, r2, x_vals, y_pred, fig, ax = result
        else:
            self.fail(
                "run_linear_regression should return at least (slope, intercept, r2, x_vals, y_pred, fig, ax)."
            )

        # Reference solution from linregress + r2_score
        lr = linregress(df["x"], df["y"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["x"]
        expected_r2 = r2_score(df["y"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should match linregress result ({expected_slope}). Got {slope}.",
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-1),
            f"Intercept should match linregress result ({expected_intercept}). Got {intercept}.",
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should match r2_score ({expected_r2}). Got {r2}.",
        )

        # Axes / labeling checks (same expectations as other regression tests)
        self.assertIsInstance(
            ax, Axes, "run_linear_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertEqual(
            "x",
            xlabel.lower(),
            "x-axis label for run_linear_regression should be exactly 'x'.",
        )
        self.assertEqual(
            "y",
            ylabel.lower(),
            "y-axis label for run_linear_regression should be exactly 'y'.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "run_linear_regression plots should have a non-empty title.",
        )

    @case_options(
        1,
        failure=(
            "Function 'run_linear_regression' fails on unsorted noisy negative data "
            "compared to SciPy linregress + r2_score"
        ),
        error="Error occurred while testing 'run_linear_regression' on unsorted noisy data",
    )
    def test_run_linear_regression_unsorted_noisy_complex(self):
        """More complex case with unsorted x-values and negative slope.

        This test checks that the implementation does not assume sorted input
        and still produces correct regression parameters and a sensible
        plotting domain.
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        # Start with a simple negative-slope relationship and add noise
        x_vals_input = np.array([0, 5, 10, 15, 20, 25, 30])
        y_clean = 50 - 1.5 * x_vals_input
        noise = np.array([0.2, -0.3, 0.1, -0.15, 0.25, -0.05, 0.0])
        y_vals_input = y_clean + noise

        # Shuffle the data to make x unsorted
        rng = np.random.default_rng(123)
        indices = rng.permutation(len(x_vals_input))
        x_shuffled = x_vals_input[indices]
        y_shuffled = y_vals_input[indices]

        df = pd.DataFrame({"x": x_shuffled, "y": y_shuffled})

        result = fn(df, "x", "y", show_plot=False)
        if isinstance(result, tuple) and len(result) >= 7:
            slope, intercept, r2, x_vals, y_pred, fig, ax = result
        else:
            self.fail(
                "run_linear_regression should return at least (slope, intercept, r2, x_vals, y_pred, fig, ax)."
            )

        # Reference solution from linregress + r2_score on the same (shuffled) data
        lr = linregress(df["x"], df["y"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["x"]
        expected_r2 = r2_score(df["y"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should match linregress result ({expected_slope}). Got {slope}.",
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-1),
            f"Intercept should match linregress result ({expected_intercept}). Got {intercept}.",
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should match r2_score ({expected_r2}). Got {r2}.",
        )

        # We expect the plotting domain (x_vals) to be sorted from min to max
        self.assertTrue(
            all(x_vals[i] <= x_vals[i + 1] for i in range(len(x_vals) - 1)),
            "x_vals returned by run_linear_regression should be in non-decreasing order for plotting.",
        )
        self.assertLessEqual(
            x_vals[0], np.min(x_vals_input),
            "x_vals should start at or before the minimum input x for a reasonable plot domain.",
        )
        self.assertGreaterEqual(
            x_vals[-1], np.max(x_vals_input),
            "x_vals should end at or after the maximum input x for a reasonable plot domain.",
        )

        # Axes / labeling checks
        self.assertIsInstance(
            ax, Axes, "run_linear_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertEqual(
            "x",
            xlabel.lower(),
            "x-axis label for run_linear_regression should be exactly 'x'.",
        )
        self.assertEqual(
            "y",
            ylabel.lower(),
            "y-axis label for run_linear_regression should be exactly 'y'.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "run_linear_regression plots should have a non-empty title.",
        )

    @case_options(
        1,
        failure="Function 'run_linear_regression' does not handle negative slopes correctly",
        error="Error occurred while testing 'run_linear_regression' with negative slope",
    )
    def test_run_linear_regression_perfect_negative_slope(self):
        """
        Simple case: perfect line with negative slope.
        y = 10 - 2x for x in {0, 1, 2, 3}
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        df = pd.DataFrame(
            {
                "x": [0, 1, 2, 3],
                "y": [10, 8, 6, 4],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(
            df, "x", "y", show_plot=False)

        self.assertTrue(
            np.isclose(r2, 1.0, atol=0.01),
            "R^2 should be 1.0 (within tolerance) for a perfect decreasing linear relationship.",
        )
        self.assertLess(
            slope, 0, "Slope should be negative for a decreasing linear trend."
        )

        expected_slope = -2.0
        expected_intercept = 10.0
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))

    @case_options(
        1,
        failure="Plot from 'run_linear_regression' is missing basic labels or title",
        error="Error occurred while checking plotting for 'run_linear_regression'",
    )
    def test_run_linear_regression_plot_formatting(self):
        """
        Check that run_linear_regression produces a plot with labeled axes and a title
        when it returns an Axes object.
        """
        fn = self.global_vars.get("run_linear_regression")
        if not callable(fn):
            self.fail(
                "Function 'run_linear_regression' is not defined or not callable in the student script."
            )

        df = pd.DataFrame(
            {
                "x": [0, 1, 2, 3],
                "y": [0, 2, 4, 6],
            }
        )
        title = "Basic Linear Regression"

        _, _, _, x_vals, y_pred, fig, ax = fn(
            df, "x", "y", title, show_plot=False)

        if ax is None:
            self.skipTest(
                "run_linear_regression does not return an Axes object; skipping plotting checks."
            )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        output_title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            isinstance(xlabel, str) and len(xlabel.strip()) > 0,
            "x-axis must have a non-empty label.",
        )
        self.assertTrue(
            xlabel == "x",
            f"Got xlabel: {xlabel}, but should be 'x'"
        )
        self.assertTrue(
            isinstance(ylabel, str) and len(ylabel.strip()) > 0,
            "y-axis must have a non-empty label.",
        )
        self.assertTrue(
            ylabel == "y",
            f"Got ylabel: {ylabel}, but should be 'y'"
        )
        self.assertTrue(
            isinstance(output_title, str) and len(output_title.strip()) > 0,
            "Plot must have a non-empty title.",
        )
        self.assertTrue(
            output_title == title,
            f"Got title: {output_title}, but should be {title}"
        )

@testsuite_options(10, 3)
class TestPart5_1(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}

    @case_options(
        1,
        failure="Function 'global_per_capita_regression' did not correctly compute regression",
        error="Error occurred while testing 'global_per_capita_regression'",
    )
    def test_global_per_capita_regression_linear_trend(self):
        """
        Perfectly linear per-capita trend → R² should be 1.
        """
        fn = self.global_vars.get("global_per_capita_regression")
        if not callable(fn):
            self.fail(
                "Function 'global_per_capita_regression' is not defined or not callable."
            )

        merged_df = pd.DataFrame(
            {
                "Year": [2000, 2001, 2002, 2003],
                "PeopleAffected": [100, 200, 300, 400],
                "Population": [10, 10, 10, 10],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(
            merged_df, show_plot=False)

        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(r2, float)
        self.assertTrue(
            np.isclose(r2, 1.0, atol=0.01),
            "R^2 should be 1.0 for a perfectly linear per-capita trend.",
        )

        expected_slope = 1000000
        expected_intercept = -1999000000
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))

    @case_options(
        1,
        failure="Function 'global_per_capita_regression' may not properly handle repeated years",
        error="Error occurred while testing global_per_capita_regression",
    )
    def test_global_per_capita_regression_handles_multiple_rows_per_year(self):
        fn = self.global_vars.get("global_per_capita_regression")
        if not callable(fn):
            self.fail("global_per_capita_regression not callable.")

        merged_df = pd.DataFrame(
            {
                "Year": [2000, 2000, 2001, 2001],
                "PeopleAffected": [50, 150, 200, 400],
                "Population": [5, 5, 10, 10],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(
            merged_df, show_plot=False)

        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(r2, float)

        expected_slope = 1000000
        expected_intercept = -1998000000
        expected_r2 = 1
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))
        self.assertTrue(np.isclose(r2, expected_r2, atol=0.01))

    @case_options(
        1,
        failure="Function 'global_per_capita_regression' fails on simple per-capita trend (template 1)",
        error="Error in 'global_per_capita_regression' template 1",
    )
    def test_global_per_capita_regression1(self):
        """
        simple linear per-capita increase over time.
        Uses linregress + r2_score on per-capita values as reference.
        """
        fn = self.global_vars.get("global_per_capita_regression")
        if not callable(fn):
            self.fail(
                "Function 'global_per_capita_regression' is not defined or not callable."
            )

        merged_df = pd.DataFrame(
            {
                "Year": [2000, 2001, 2002, 2003],
                "PeopleAffected": [100, 150, 200, 250],
                "Population": [10, 10, 10, 10],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(merged_df, show_plot=False)

        # reference: per-capita per 100k or whatever scaling you're using
        per_capita = merged_df["PeopleAffected"] / merged_df["Population"] * 100000
        lr = linregress(merged_df["Year"], per_capita)
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * merged_df["Year"]
        expected_r2 = r2_score(per_capita, expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should be close to linregress result on per-capita ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should be close to linregress result on per-capita ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should be close to r2_score result on per-capita ({expected_r2}). Got {r2}."
        )

        # Axes / labeling checks
        self.assertIsInstance(
            ax, Axes, "global_per_capita_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            len(xlabel.strip()) > 0,
            "x-axis label for global_per_capita_regression should be non-empty.",
        )
        self.assertEqual(
            "year",
            xlabel.lower(),
            "x-axis label for global_per_capita_regression should be 'Year'.",
        )

        self.assertTrue(
            len(ylabel.strip()) > 0,
            "y-axis label for global_per_capita_regression should be non-empty.",
        )
        self.assertEqual(
            "affected_per_100k",
            ylabel.lower(),
            "y-axis label for global_per_capita_regression should be 'affected_per_100k'.",
        )

        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "global_per_capita_regression plots should have a non-empty title.",
        )
        self.assertEqual(
            "people affected per 100k (global)",
            title.lower(),
            f"title for global_per_capita_regression should be 'People Affected per 100k (Global)', got: {title}.",
        )

    @case_options(
        1,
        failure="Function 'global_per_capita_regression' fails on varying population (template 2)",
        error="Error in 'global_per_capita_regression' template 2",
    )
    def test_global_per_capita_regression2(self):
        """
        Linear trend in people affected with changing population.
        Uses linregress + r2_score on per-capita as reference.
        """
        fn = self.global_vars.get("global_per_capita_regression")
        if not callable(fn):
            self.fail(
                "Function 'global_per_capita_regression' is not defined or not callable."
            )

        merged_df = pd.DataFrame(
            {
                "Year": [2000, 2001, 2002, 2003],
                "PeopleAffected": [100, 180, 260, 340],
                "Population": [10, 12, 14, 16],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(merged_df, show_plot=False)

        per_capita = merged_df["PeopleAffected"] / merged_df["Population"] * 100000
        lr = linregress(merged_df["Year"], per_capita)
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * merged_df["Year"]
        expected_r2 = r2_score(per_capita, expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-2),
            f"Slope should match linregress on per-capita ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should match linregress on per-capita ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should match r2_score on per-capita ({expected_r2}). Got {r2}."
        )

        # Axes / labeling checks
        self.assertIsInstance(
            ax, Axes, "global_per_capita_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            len(xlabel.strip()) > 0,
            "x-axis label for global_per_capita_regression should be non-empty.",
        )
        self.assertEqual(
            "year",
            xlabel.lower(),
            "x-axis label for global_per_capita_regression should be 'Year'.",
        )

        self.assertTrue(
            len(ylabel.strip()) > 0,
            "y-axis label for global_per_capita_regression should be non-empty.",
        )
        self.assertEqual(
            "affected_per_100k",
            ylabel.lower(),
            "y-axis label for global_per_capita_regression should be 'affected_per_100k'.",
        )

        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "global_per_capita_regression plots should have a non-empty title.",
        )
        self.assertEqual(
            "people affected per 100k (global)",
            title.lower(),
            f"title for global_per_capita_regression should be 'People Affected per 100k (Global)', got: {title}.",
        )

    @case_options(
        1,
        failure="Plot from 'global_per_capita_regression' is missing basic labels or title",
        error="Error occurred while checking plotting for 'global_per_capita_regression'",
    )
    def test_global_per_capita_regression_plot_formatting(self):
        """
        Check that global_per_capita_regression produces a plot with labeled axes and a title
        when it returns an Axes object.
        """
        fn = self.global_vars.get("global_per_capita_regression")
        if not callable(fn):
            self.fail(
                "Function 'global_per_capita_regression' is not defined or not callable in the student script."
            )

        merged_df = pd.DataFrame(
            {
                "Year": [2000, 2001, 2002, 2003],
                "PeopleAffected": [100, 200, 300, 400],
                "Population": [10, 10, 10, 10],
            }
        )

        _, _, _, x_vals, y_pred, fig, ax = fn(merged_df, show_plot=False)

        if ax is None:
            self.skipTest(
                "global_per_capita_regression does not return an Axes object; skipping plotting checks."
            )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        output_title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            isinstance(xlabel, str) and len(xlabel.strip()) > 0,
            "x-axis must have a non-empty label.",
        )
        self.assertTrue(
            xlabel == "Year",
            f"Got xlabel: {xlabel}, but should be 'Year'"
        )
        self.assertTrue(
            isinstance(ylabel, str) and len(ylabel.strip()) > 0,
            "y-axis must have a non-empty label.",
        )
        self.assertTrue(
            ylabel.lower() == "affected_per_100k",
            f"Got ylabel: {ylabel}, but should be 'affected_per_100k'"
        )
        self.assertTrue(
            isinstance(output_title, str) and len(output_title.strip()) > 0,
            "Plot must have a non-empty title.",
        )
        self.assertTrue(
            output_title.lower() == "people affected per 100k (global)",
            f"Got title: {output_title}, but should be 'People Affected per 100k (Global)'"
        )

@testsuite_options(10, 3)
class TestPart5_2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}

    @case_options(
        1,
        failure="Function 'temp_anomaly_regression' did not capture a positive temperature trend",
        error="Error occurred while testing monotonic increasing temps",
    )
    def test_temp_anomaly_regression_monotonic_increasing(self):
        fn = self.global_vars.get("temp_anomaly_regression")
        if not callable(fn):
            self.fail("temp_anomaly_regression not callable.")

        df = pd.DataFrame(
            {
                "Year": [1980, 1990, 2000, 2010],
                "Temp": [0.1, 0.2, 0.3, 0.4],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(df, show_plot=False)

        self.assertGreater(slope, 0, "Slope should be positive.")
        self.assertGreater(r2, 0.9, "R² should be high for a linear trend.")

        expected_slope = 0.01
        expected_intercept = -19.7
        expected_r2 = 1.0
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))
        self.assertTrue(np.isclose(r2, expected_r2, atol=0.01))

    @case_options(
        1,
        failure="Function 'temp_anomaly_regression' fails on simple warming trend",
        error="Error occurred while testing 'temp_anomaly_regression'",
    )
    def test_temp_anomaly_regression1(self):
        """
        simple linear warming.
        """
        fn = self.global_vars.get("temp_anomaly_regression")
        if not callable(fn):
            self.fail("temp_anomaly_regression not callable.")

        df = pd.DataFrame(
            {
                "Year": [1980, 1990, 2000, 2010],
                "Temp": [0.0, 0.1, 0.2, 0.3],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(df, show_plot=False)

        lr = linregress(df["Year"], df["Temp"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["Year"]
        expected_r2 = r2_score(df["Temp"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-3),
            f"Slope should be close to linregress result ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should be close to linregress result ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should match r2_score ({expected_r2}). Got {r2}."
        )

        # Axes / labeling checks
        self.assertIsInstance(
            ax, Axes, "temp_anomaly_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertEqual(
            "year",
            xlabel.lower(),
            "x-axis label for temp_anomaly_regression should be 'Year'.",
        )
        self.assertTrue(
            "temp" == ylabel.lower(),
            "y-axis label for temp_anomaly_regression should be 'Temp'.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "temp_anomaly_regression plots should have a non-empty title.",
        )
        self.assertEqual(
            "global temperature anomaly regression",
            title.lower(),
            f"title for temp_anomaly_regression should be 'Global Temperature Anomaly Regression', got: {title}.",
        )

    @case_options(
        1,
        failure="Function 'temp_anomaly_regression' did not correctly average duplicate years",
        error="Error while testing 'temp_anomaly_regression'",
    )
    def test_temp_anomaly_regression_averages_over_duplicate_years(self):
        fn = self.global_vars.get("temp_anomaly_regression")
        if not callable(fn):
            self.fail("temp_anomaly_regression not callable.")

        df = pd.DataFrame(
            {
                "Year": [2000, 2000, 2001, 2001, 2002, 2002],
                "Temp": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(df, show_plot=False)

        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(r2, float)
        self.assertTrue(
            np.isclose(r2, 1.0, atol=0.01),
            "Averaged anomalies are perfectly linear, so R² should be 1.",
        )

        expected_slope = 0.4
        expected_intercept = -799.9
        self.assertTrue(np.isclose(slope, expected_slope, atol=0.01))
        self.assertTrue(np.isclose(intercept, expected_intercept, atol=0.01))

    @case_options(
        1,
        failure="Function 'temp_anomaly_regression' fails on noisy warming trend (template 2)",
        error="Error occurred while testing 'temp_anomaly_regression' template 2",
    )
    def test_temp_anomaly_regression2(self):
        """
        Testing "temp_anomaly_regression" on noisy
        but increasing temperatures.
        """
        fn = self.global_vars.get("temp_anomaly_regression")
        if not callable(fn):
            self.fail("temp_anomaly_regression not callable.")

        df = pd.DataFrame(
            {
                "Year": [1980, 1985, 1990, 1995, 2000],
                "Temp": [0.0, 0.05, 0.12, 0.18, 0.28],
            }
        )

        slope, intercept, r2, x_vals, y_pred, fig, ax = fn(df, show_plot=False)

        lr = linregress(df["Year"], df["Temp"])
        expected_slope = lr.slope
        expected_intercept = lr.intercept
        expected_y_pred = lr.intercept + lr.slope * df["Year"]
        expected_r2 = r2_score(df["Temp"], expected_y_pred)

        self.assertTrue(
            np.isclose(slope, expected_slope, atol=1e-3),
            f"Slope should be close to linregress result ({expected_slope}). Got {slope}."
        )
        self.assertTrue(
            np.isclose(intercept, expected_intercept, atol=1e-2),
            f"Intercept should be close to linregress result ({expected_intercept}). Got {intercept}."
        )
        self.assertTrue(
            np.isclose(r2, expected_r2, atol=1e-3),
            f"R^2 should match r2_score ({expected_r2}). Got {r2}."
        )

        # Axes / labeling checks
        self.assertIsInstance(
            ax, Axes, "temp_anomaly_regression should return a Matplotlib Axes object."
        )
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertEqual(
            "year",
            xlabel.lower(),
            f"x-axis label for temp_anomaly_regression should be 'Year', got: {xlabel}.",
        )
        self.assertTrue(
            "temp" == ylabel.lower(),
            f"y-axis label for temp_anomaly_regression should be 'Temp', got: {ylabel}.",
        )
        self.assertTrue(
            isinstance(title, str) and len(title.strip()) > 0,
            "temp_anomaly_regression plots should have a non-empty title.",
        )
        self.assertEqual(
            "global temperature anomaly regression",
            title.lower(),
            f"title for temp_anomaly_regression should be 'Global Temperature Anomaly Regression', got: {title}.",
        )

    @case_options(
        1,
        failure="Plot from 'temp_anomaly_regression' is missing basic labels or title",
        error="Error occurred while checking plotting for 'temp_anomaly_regression'",
    )
    def test_temp_anomaly_regression_plot_formatting(self):
        """
        Check that temp_anomaly_regression produces a plot with labeled axes and a title
        when it returns an Axes object.
        """
        fn = self.global_vars.get("temp_anomaly_regression")
        if not callable(fn):
            self.fail(
                "Function 'temp_anomaly_regression' is not defined or not callable in the student script."
            )

        df = pd.DataFrame(
            {
                "Year": [2000, 2001, 2002],
                "Temp": [0.1, 0.2, 0.3],
            }
        )

        _, _, _, x_vals, y_pred, fig, ax = fn(df, show_plot=False)

        if ax is None:
            self.skipTest(
                "temp_anomaly_regression does not return an Axes object; skipping plotting checks."
            )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        output_title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            isinstance(xlabel, str) and len(xlabel.strip()) > 0,
            "x-axis must have a non-empty label.",
        )
        self.assertTrue(
            xlabel == "Year",
            f"Plot must have xlabel: 'Year' but got: {xlabel}."
        )
        self.assertTrue(
            isinstance(ylabel, str) and len(ylabel.strip()) > 0,
            "y-axis must have a non-empty label.",
        )
        self.assertTrue(
            ylabel == "Temp",
            f"Plot must have ylabel: 'Temp' but got: {ylabel}."
        )
        self.assertTrue(
            isinstance(output_title, str) and len(output_title.strip()) > 0,
            "Plot must have a non-empty title.",
        )
        self.assertTrue(
            output_title == "Global Temperature Anomaly Regression",
            f"Plot must have title: 'Global Temperature Anomaly Regression' but got: {output_title}."
        )

@testsuite_options(10, 4)
class TestPart5_3(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # file is in grandparent directory
        self.student_script_path = "pset.py"
        # USING THIS FOR MAKING TEST CASES BUT USE ABOVE FOR ACTUAL TEST.PY
        _, student_locals, _ = run_student_script(self.student_script_path, {})

        self.global_vars = student_locals or {}
    @case_options(
        3,
        failure="Function 'polynomial_regression_disasters' did not correctly fit polynomial models to disaster counts",
        error="Error occurred while testing 'polynomial_regression_disasters'",
    )
    def test_polynomial_regression_disasters(self):
        """
        Check that polynomial_regression_disasters returns expected coefficients,
        predictions, and R^2 for the real disasters dataset.
        """
        fn = self.global_vars.get("polynomial_regression_disasters")
        if not callable(fn):
            self.fail(
                "Function 'polynomial_regression_disasters' is not defined or not callable in the student script."
            )

        student_load_data = self.global_vars.get("load_data")
        student_proc_dis = self.global_vars.get("process_disaster_data")
        if not callable(student_load_data) or not callable(student_proc_dis):
            self.fail(
                "load_data or process_disaster_data not callable in student script.")

        # Use the same disasters file and preprocessing as the pset
        df_dis_unproc = student_load_data(DISASTERS_FILE)
        df_dis = student_proc_dis(df_dis_unproc, people=False)

        degree_to_vals, ax = fn(df_dis, show_plot=False)

        # Basic structure checks
        self.assertIsInstance(degree_to_vals, dict)
        for deg in [1, 2, 5, 10]:
            self.assertIn(
                deg,
                degree_to_vals,
                f"Degree {deg} missing from polynomial_regression_disasters return value.",
            )

        # Unpack degree 1 results: [coeffs, y_pred, r2]
        coeffs_1, y_pred_1, r2_1 = degree_to_vals[1]

        # Expected values (from reference run)
        expected_coeffs_1 = np.array([5.99591568e+00, -1.17534899e+04])
        expected_y_pred_1 = np.array(
            [
                118.42318841, 124.41910408, 130.41501976, 136.41093544,
                142.40685112, 148.4027668, 154.39868248, 160.39459816,
                166.39051383, 172.38642951, 178.38234519, 184.37826087,
                190.37417655, 196.37009223, 202.36600791, 208.36192358,
                214.35783926, 220.35375494, 226.34967062, 232.3455863,
                238.34150198, 244.33741765, 250.33333333, 256.32924901,
                262.32516469, 268.32108037, 274.31699605, 280.31291173,
                286.3088274, 292.30474308, 298.30065876, 304.29657444,
                310.29249012, 316.2884058, 322.28432148, 328.28023715,
                334.27615283, 340.27206851, 346.26798419, 352.26389987,
                358.25981555, 364.25573123, 370.2516469, 376.24756258,
                382.24347826
            ]
        )
        expected_r2_1 = 0.762261089174479

        # Shape checks
        self.assertEqual(
            coeffs_1.shape,
            expected_coeffs_1.shape,
            "Degree-1 coefficient vector has wrong shape.",
        )
        self.assertEqual(
            y_pred_1.shape,
            expected_y_pred_1.shape,
            "Degree-1 prediction array has wrong shape.",
        )

        # Numerical checks (tolerances allow for floating-point noise)
        self.assertTrue(
            np.allclose(coeffs_1, expected_coeffs_1, rtol=0.1),
            f"Degree-1 polynomial coefficients do not match expected values.\n"
            f"Got:      {coeffs_1}\nExpected: {expected_coeffs_1}",
        )
        self.assertTrue(
            np.allclose(y_pred_1, expected_y_pred_1, rtol=0.1),
            "Degree-1 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.isclose(r2_1, expected_r2_1, rtol=0.01),
            f"Degree-1 R^2 does not match expected value. Got {r2_1}, expected {expected_r2_1}.",
        )
        # Unpack degree 2 results: [coeffs, y_pred, r2]
        coeffs_2, y_pred_2, r2_2 = degree_to_vals[2]

        expected_coeffs_2 = np.array(
            [-1.53083336e-01,  6.18941595e+02, -6.25286295e+05]
        )
        expected_y_pred_2 = np.array(
            [
                70.15090965,  82.7294088,  95.00174127, 106.96790707,
                118.62790619, 129.98173865, 141.02940443, 151.77090354,
                162.20623597, 172.33540173, 182.15840082, 191.67523324,
                200.88589898, 209.79039805, 218.38873045, 226.68089618,
                234.66689523, 242.34672761, 249.72039331, 256.78789235,
                263.54922471, 270.0043904, 276.15338941, 281.99622175,
                287.53288742, 292.76338642, 297.68771874, 302.30588439,
                306.61788337, 310.62371568, 314.32338131, 317.71688027,
                320.80421255, 323.58537817, 326.06037711, 328.22920938,
                330.09187497, 331.64837389, 332.89870614, 333.84287172,
                334.48087062, 334.81270285, 334.83836841, 334.55786729,
                333.97119951
            ]
        )
        expected_r2_2 = 0.8292069314976142

        self.assertEqual(
            coeffs_2.shape,
            expected_coeffs_2.shape,
            "Degree-2 coefficient vector has wrong shape.",
        )
        self.assertEqual(
            y_pred_2.shape,
            expected_y_pred_2.shape,
            "Degree-2 prediction array has wrong shape.",
        )
        self.assertTrue(
            np.allclose(coeffs_2, expected_coeffs_2, rtol=0.1),
            f"Degree-2 polynomial coefficients do not match expected values.\n"
            f"Got:      {coeffs_2}\nExpected: {expected_coeffs_2}",
        )
        self.assertTrue(
            np.allclose(y_pred_2, expected_y_pred_2, rtol=0.1),
            "Degree-2 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.isclose(r2_2, expected_r2_2, rtol=0.01),
            f"Degree-2 R^2 does not match expected value. Got {r2_2}, expected {expected_r2_2}.",
        )

        # Unpack degree 5 results: [coeffs, y_pred, r2]
        coeffs_5, y_pred_5, r2_5 = degree_to_vals[5]

        expected_coeffs_5 = np.array([2.16754599e-05, -2.16007219e-01,  8.61019482e+02, -1.71598091e+06,
             1.70988406e+09, -6.81499561e+11])

        expected_y_pred_5 = np.array(
            [
                107.296875, 106.17211914, 106.63195801, 108.67163086,
                112.25634766, 117.32446289, 123.79333496, 131.5546875,
                140.48730469, 150.44995117, 161.28869629, 172.84020996,
                184.93334961, 197.39050293, 210.03015137, 222.6730957,
                235.14050293, 247.26037598, 258.8626709, 269.79846191,
                279.91967773, 289.10021973, 297.23144531, 304.22265625,
                310.00952148, 314.55102539, 317.83459473, 319.88061523,
                320.74060059, 320.50524902, 319.29931641, 317.29382324,
                314.70141602, 311.7800293, 308.83935547, 306.23828125,
                304.3939209, 303.77648926, 304.91699219, 308.40795898,
                314.90869141, 325.14489746, 339.91040039, 360.07434082,
                386.57836914
            ]
        )
        expected_r2_5 = 0.8761889824079836

        self.assertEqual(
            y_pred_5.shape,
            expected_y_pred_5.shape,
            "Degree-5 prediction array has wrong shape.",
        )
        self.assertTrue(
            np.allclose(coeffs_5, expected_coeffs_5, rtol=1),
            "Degree-5 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.allclose(y_pred_5, expected_y_pred_5, rtol=0.1),
            "Degree-5 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.isclose(r2_5, expected_r2_5, rtol=0.1),
            f"Degree-5 R^2 does not match expected value. Got {r2_5}, expected {expected_r2_5}.",
        )

        # Unpack degree 10 results: [coeffs, y_pred, r2]
        coeffs_10, y_pred_10, r2_10 = degree_to_vals[10]

        expected_coeffs_10 = np.array(
            [-3.42093993e-20,  2.19180046e-16, -2.64921333e-13, -6.59710522e-10,
             4.38085007e-07,  2.94181384e-03,  1.77257218e+00, -1.06254200e+04,
             -1.70995954e+07,  5.67147407e+10, -3.54913283e+13]
        )
        expected_y_pred_10 = np.array(
            [
                73.3203125, 104.4296875, 122.171875, 130.8515625, 133.8515625,
                133.9375, 133.125, 133.015625, 134.625, 138.6953125,
                145.4453125, 154.9765625, 166.953125, 181.203125, 197.09375,
                214.0703125, 231.5390625, 248.8515625, 265.375, 280.6328125,
                294.0703125, 305.421875, 314.328125, 320.671875, 324.421875,
                325.703125, 324.7265625, 321.84375, 317.4765625, 312.1796875,
                306.5078125, 301.1171875, 296.59375, 293.59375, 292.5859375,
                294.015625, 298.1328125, 304.890625, 314.09375, 325.0859375,
                336.8359375, 347.859375, 356.015625, 358.5859375, 352.
            ]
        )
        expected_r2_10 = 0.9049687285548651

        self.assertEqual(
            coeffs_10.shape,
            expected_coeffs_10.shape,
            "Degree-10 coefficient vector has wrong shape.",
        )
        self.assertEqual(
            y_pred_10.shape,
            expected_y_pred_10.shape,
            "Degree-10 prediction array has wrong shape.",
        )
        self.assertTrue(
            np.allclose(coeffs_10, expected_coeffs_10, rtol=1),
            "Degree-5 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.allclose(y_pred_10, expected_y_pred_10, rtol=0.1),
            "Degree-10 polynomial predictions do not match expected values.",
        )
        self.assertTrue(
            np.isclose(r2_10, expected_r2_10, rtol=0.1),
            f"Degree-10 R^2 does not match expected value. Got {r2_10}, expected {expected_r2_10}.",
        )

        self.assertIsInstance(
            ax, Axes, "polynomial_regression_disasters should return a Matplotlib Axes as the second value."
        )

    @case_options(
        1,
        failure="Function 'polynomial_regression_disasters' did not correctly plot the polynomial models to disaster counts",
        error="Error occurred while testing the plot formatting of 'polynomial_regression_disasters'",
    )
    def test_polynomial_regression_disasters_plot_formatting(self):
        """
        Check that polynomial_regression_disasters produces a plot with labeled axes and a title
        when it returns an Axes object.
        """
        fn = self.global_vars.get("polynomial_regression_disasters")
        if not callable(fn):
            self.fail(
                "Function 'polynomial_regression_disasters' is not defined or not callable in the student script."
            )

        student_load_data = self.global_vars.get("load_data")
        student_proc_dis = self.global_vars.get("process_disaster_data")
        if not callable(student_load_data) or not callable(student_proc_dis):
            self.fail(
                "load_data or process_disaster_data not callable in student script.")

        # Use the same disasters file and preprocessing as the pset
        df_dis_unproc = student_load_data(DISASTERS_FILE)
        df_dis = student_proc_dis(df_dis_unproc, people=False)

        _, ax = fn(df_dis, show_plot=False)

        if ax is None:
            self.skipTest(
                "temp_anomaly_regression does not return an Axes object; skipping plotting checks."
            )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        output_title = ax.get_title()
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "Plot must contain a legend.")
        self.assertTrue(
            isinstance(xlabel, str) and len(xlabel.strip()) > 0,
            "x-axis must have a non-empty label.",
        )
        self.assertTrue(
            xlabel == "Year",
            f"Plot must have ylabel: 'Year' but got: {xlabel}."
        )
        self.assertTrue(
            isinstance(ylabel, str) and len(ylabel.strip()) > 0,
            "y-axis must have a non-empty label.",
        )
        self.assertTrue(
            ylabel == "Total Number of Disasters",
            f"Plot must have ylabel: 'Total Number of Disasters' but got: {ylabel}."
        )
        self.assertTrue(
            isinstance(output_title, str) and len(output_title.strip()) > 0,
            "Plot must have a non-empty title.",
        )
        self.assertTrue(
            output_title == "Polynomial Regression on Yearly Global Disasters",
            f"Plot must have a title: 'Polynomial Regression on Yearly Global Disasters' but got: {output_title}.",
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

if __name__ == "__main__":
    test_parts = [
        TestPart2,
        TestPart3,
        TestPart4,
        TestPart5,
        TestPart5_1,
        TestPart5_2,
        TestPart5_3
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

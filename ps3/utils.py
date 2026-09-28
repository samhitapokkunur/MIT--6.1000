# Standard library
import json
import os
import random

# Third-party
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


COLORS = [
    "#ff7f0e",  # orange
    "#1f77b4",  # dark blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow
    "#17becf",  # light blue
]


def read_data_from_file(file_path, shuffle_order=False):
    """
    Reads the data from a JSON file and returns the relevant components.

    Parameters:
        file_path (str): The path to the JSON file.
        shuffle_order (boolean): Indicates if town order should be shuffled.

    Returns a tuple containing:
        filepath (str): The path to the edge file.
        num_districts (int): The number of districts.
        town_populations (dict): A dictionary mapping town names (str) to their populations (int).
        voter_party1_proportions (dict): A dictionary mapping town names
            to their voter party 1 proportions (float).
    """
    base_dir = os.path.join(os.path.dirname(__file__))
    with open(f"{base_dir}/{file_path}", "r") as f:
        data = json.load(f)
    num_districts = data["num_districts"]

    town_populations = {
        name: pop
        for name, pop in zip(data["town_names"], data["town_populations"])
    }
    voter_party1_proportions = {
        name: prop
        for name, prop in zip(
            data["town_names"], data["voter_party1_proportions"]
        )
    }

    if shuffle_order:
        items = list(town_populations.items())
        random.shuffle(items)
        town_populations = dict(items)

        items = list(voter_party1_proportions.items())
        random.shuffle(items)
        voter_party1_proportions = dict(items)

    filepath = data["edge_filepath"]
    return filepath, num_districts, town_populations, voter_party1_proportions


def voronoi_finite_polygons_2d(vor, radius=1000):
    """
    Reconstructs infinite Voronoi regions in a 2D diagram to finite regions.

    Parameters:
        vor (Voronoi): A Voronoi object representing the input diagram.
        radius (float): An optional value measuring distance to 'points at infinity'.

    Returns a tuple containing:
        regions (list): Indices of vertices in each revised Voronoi region.
        vertices (ndarray): Coordinates for revised vertices.

    Adapted from: https://gist.github.com/pv/8036995
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map ridge vertices to ridges for each point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct each region.
    # The regions will be returned in the same order as vor.points.
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region.
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue

            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region vertices counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_graph(
    graph,
    partition,
    file_path,
    title="State Partition Graph",
    ax=None,
    show_plot=True,
):
    """
    Plots the given graph using networkx and matplotlib.

    Parameters:
        graph (dict): A dictionary mapping town names to an adjacency list.
        partition (list): A list of districts,
            where each district is represented as a list of towns names.
        file_path (str): Path/to/filename of data.
        title (str): Plot title.
        ax (Axes or array): Axes of the graph.
        show_plot (boolean): Indicates if plot should be displayed.
    """
    base_dir = os.path.join(os.path.dirname(__file__))
    with open(f"{base_dir}/{file_path}", "r") as f:
        data = json.load(f)
    node_locations = data.get("town_graph_positions", {})

    G = nx.Graph()

    # Add nodes
    node_colors = {}
    for node in graph.keys():
        G.add_node(node)

    for i, district in enumerate(partition):
        for town in district:
            # Use the provided 'colors' argument, which is a tuple of colors
            node_colors[town] = COLORS[i]

    # Add edges with weights
    for src in graph.keys():
        for dest in graph[src]:
            G.add_edge(src, dest)

    # Draw nodes and edges
    colors = [node_colors.get(node, "skyblue") for node in G.nodes()]

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(
        G,
        node_locations,
        ax=ax,
        with_labels=True,
        node_color=colors,
        node_size=300,
        font_size=12,
        font_weight="bold",
    )
    edge_labels = nx.get_edge_attributes(G, "")

    nx.draw_networkx_edge_labels(
        G, node_locations, edge_labels=edge_labels, font_size=12
    )
    ax.set_title(title, size=20)
    if show_plot:
        plt.show()


def plot_voronoi_from_graph(
    graph, partition, file_path, ax=None, show_plot=True
):
    """
    Creates Voronoi diagram from planar graph.

    Parameters:
        graph (dict): A dictionary mapping town names to an adjacency list.
        partition (list): A list of districts,
            where each district is represented as a list of towns names.
        file_path (str): Path/to/filename of data.
        ax (Axes or array): Axes of the graph.
        show_plot (boolean): Indicates if plot should be displayed.
    """
    base_dir = os.path.join(os.path.dirname(__file__))
    with open(f"{base_dir}/{file_path}", "r") as f:
        data = json.load(f)
    center_points = data.get("town_graph_positions", {})

    G = nx.Graph(graph)

    node_list = list(G.nodes())
    coords = np.array([center_points[node] for node in node_list])
    vor = Voronoi(coords)

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Get the colorings
    color_map = {town: "white" for town in center_points.keys()}
    for i, district in enumerate(partition):
        for town in district:
            color_map[town] = COLORS[i]

    regions, vertices = voronoi_finite_polygons_2d(vor, radius=1000)

    # Fill each region with its corresponding color.
    for i, region in enumerate(regions):
        polygon = vertices[region]
        town = node_list[i]
        ax.fill(
            *zip(*polygon), color=color_map[town], alpha=0.4, edgecolor="grey"
        )

    # Draw the original Voronoi edges in light grey
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=False,
        show_vertices=False,
        line_colors="grey",
        line_alpha=0.5,
        line_width=1,
    )

    for town_name, (x, y) in center_points.items():
        ax.text(
            x,
            y,
            town_name,
            color="k",
            ha="center",
            va="center",
        )

    plt.xticks([])
    plt.yticks([])
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Voronoi Map", size=20)
    plt.tight_layout()

    if show_plot:
        plt.show()


def plot_graph_and_voronoi(graph, partition, file_path, show_plot=True):
    """
    Converts a planar graph (given as an adjacency dict) to a Voronoi diagram.

    Parameters:
        graph (dict): A dictionary mapping town names to an adjacency list.
        partition (list): A list of districts,
            where each district is represented as a list of towns names.
        file_path (str): Path/to/filename of data.
        show_plot (boolean): Indicates if the plot should be displayed.
    """
    # Load in the center points from the JSON file

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot_graph(graph, partition, file_path, ax=axs[0], show_plot=False)
    plot_voronoi_from_graph(graph, partition, file_path, ax=axs[1])


if __name__ == "__main__":
    pass

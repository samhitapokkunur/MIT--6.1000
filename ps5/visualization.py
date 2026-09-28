import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class EnvVisualization:
    def __init__(self, env):
        self.env = env

    def render(self, time_step):
        raise NotImplementedError()


class TaxiEnvVisualizaton(EnvVisualization):
    def __init__(self, env):
        super().__init__(env)

    def _init_render(self):
        # initialize plot
        self._fig, self._ax = plt.subplots()
        self._ax.invert_yaxis()
        self._ax.set_xlim(-0.5, self.env.width + 1 - 0.5)
        self._ax.set_ylim(-0.5, self.env.height + 1 - 0.5)
        self._ax.set_xticks(np.arange(-0.5, self.env.width + 1, 1))
        self._ax.set_yticks(np.arange(-0.5, self.env.height + 1, 1))
        self._ax.set_xticklabels([])
        self._ax.set_yticklabels([])
        self._ax.grid(True)
        self._ax.set_aspect("equal")

        # initialize taxi, passenger, and destination
        (self.taxi_marker,) = self._ax.plot(
            *self.env.taxi_location, "ys", markersize=20, label="Taxi"
        )
        (self.passenger_marker,) = self._ax.plot(
            *self.env.passenger_location, "bo", markersize=12, label="Passenger"
        )
        (self.dest_marker,) = self._ax.plot(
            *self.env.destination_location, "r*", markersize=15, label="Destination"
        )

        self._ax.legend(
            loc="center left",  # place to the left of the plot
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            fontsize="small",
            frameon=False,
            labelspacing=1.3,
            handlelength=1.5,
            handletextpad=0.6,
        )

        self._ax.text(
            0,
            0,
            "(0,0)",
            fontsize=9,
            color="gray",
            ha="left",
            va="bottom",
            fontweight="bold",
        )

        self._fig.tight_layout(rect=[0, 0, 0.85, 1])
        plt.ion()
        plt.show()

    def _draw_agents(self):
        # Update markers
        px, py = self.env.passenger_location
        dx, dy = self.env.destination_location
        tx, ty = self.env.taxi_location

        # update marker positions
        self.taxi_marker.set_data([tx], [ty])
        if not self.env.picked_up_passenger:
            self.passenger_marker.set_data([px], [py])
            self.passenger_marker.set_visible(True)
        else:
            self.passenger_marker.set_visible(False)

        self.dest_marker.set_data([dx], [dy])

    def render(self, time_step):
        if time_step == 1:
            self._init_render()
        else:
            self._draw_agents()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.5)  # short pause to let GUI update


class FrozenTaxiEnvVisualization(TaxiEnvVisualizaton):
    def __init__(self, env):
        super().__init__(env)

    def _draw_boulders(self):
        # draw boulders
        bx, by = zip(*self.env.boulders)
        self._ax.plot(
            bx, by, "s", color="saddlebrown", markersize=20, label="Boulder"
        )

    def render(self, time_step):
        if time_step == 1:
            self._init_render()
            self._draw_boulders()
        else:
            self._draw_agents()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.5)  # short pause to let GUI update


class DiseaseSimVisualization(EnvVisualization):
    '''
    Diseas Simulation Visualization
    ________________________________________
    |                    |                  |
    |                    |     Graph of     |
    | disease simulation | Population Stats |
    |                    |__________________|
    |                    | Population Stats |
    |                    |      Text        |
    |____________________|__________________|

    '''
    def __init__(self, env):
        super().__init__(env)

        # agent colors
        self.state_colors = {
            "healthy": "green",
            "infected": "red",
            "recovered": "blue",
            "inactive": "gray",
        }

        # agent shapes
        self.agent_shapes = {
            "random": "s",  # square
            "careful": "o",  # circle
            "menacing": "^",  # triangle
        }

    ### helper funtions:

    def _get_agent_color(self, info):
        if not info["active"]:
            return self.state_colors["inactive"]
        elif info["infected"]:
            return self.state_colors["infected"]
        else:
            return self.state_colors["healthy"]
        
    def _init_sim(self):
        # Helper function for initializing sim
        self._ax_sim.set_title("Disease Simulation")
        self._ax_sim.set_xlim(-0.5, self.env.width + 1 + 0.5)
        self._ax_sim.set_ylim(-0.5, self.env.height + 1 + 0.5)
        self._ax_sim.set_aspect("equal")
        self._ax_sim.set_xticks(np.arange(-0.5, self.env.width + 1 + 0.5, 10))
        self._ax_sim.set_yticks(np.arange(-0.5, self.env.height + 1 + 0.5, 10))
        self._ax_sim.set_xticklabels([])
        self._ax_sim.set_yticklabels([])
        self._ax_sim.grid(True, linestyle="--", alpha=0.1)
        self._ax_sim.set_aspect("equal")
        
        # Legend
        shape_handles = [
            lines.Line2D(
                [],
                [],
                color="black",
                marker=shape,
                linestyle="None",
                markersize=8,
                label=atype.capitalize(),
            )
            for atype, shape in self.agent_shapes.items()
        ]
        # Infection status colors
        infection_handles = [
            patches.Patch(color="red", alpha=0.6, label="Infected"),
            patches.Patch(color="green", alpha=0.6, label="Healthy"),
            patches.Patch(color="gray", alpha=0.6, label="Inactive"),
        ]
        legend_handles = shape_handles + infection_handles
        self._ax_sim.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            fontsize=7,
            frameon=False,
            labelspacing=0.8,
        )

    def _init_graph(self):
        # Helper function for initializing Graph
        self.max_num_steps = 100
        self._ax_graph.set_xlim(0, self.max_num_steps)
        self._ax_graph.set_ylim(0, 100)
        self._ax_graph.set_title("Population Status Over Time")
        self._ax_graph.set_xlabel("Time step")
        self._ax_graph.set_ylabel("Percentage of People (%)")

        self.infected_percentages = []
        self.dead_percentages = []
        (self.line_infected,) = self._ax_graph.plot([], [], label="Infected", color="orange")
        (self.line_dead,) = self._ax_graph.plot([], [], label="Dead", color="red")
        self._ax_graph.legend()

    def _init_info(self):
        # helper funciton for information box
        self._ax_info.axis("off")
        self.info_text = self._ax_info.text(
            0.25, 0.75, "",
            va="top", ha="left", fontsize=11, family="monospace",
            color="#2c3e50",
            linespacing=1.4, 
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f9f9", edgecolor="#bdc3c7")
        )
        self._ax_info.set_title("Simulation Info", fontsize=12, pad=10)


    def _init_render(self):
        # format figure and subplots
        self._fig = plt.figure(figsize=(12, 6.5))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[2.3, 1],    # left = sim, right = analytics
            height_ratios=[1.2, 0.6], # top (graph) taller
            wspace=0.6,              # extra horizontal padding
            hspace=0.75               # slightly tighter vertical spacing
        )

        # add subplot for sim on left, graph on top right, info text on bototm right
        self._ax_sim = self._fig.add_subplot(gs[:, 0])      # spans both rows (left)
        self._ax_graph = self._fig.add_subplot(gs[0, 1])    # top-right
        self._ax_info = self._fig.add_subplot(gs[1, 1])     # bottom-right

        # initiate sim, graph, and text box
        self._init_sim()
        self._init_graph()
        self._init_info()

        # initialize agents
        self.scatter_artists = {}  # {shape: PathCollection}
        self.radius_patches = []  # contagious radius circles
        self._draw_agents(initial=True)

        # timestep label
        self._time_text = self._ax_sim.text(
            0.02,
            0.98,
            f"t = 0",
            transform=self._ax_sim.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        self._fig.tight_layout()
        plt.ion()
        plt.show()

    def _draw_agents(self, initial=False):
        infos = self.env.infos
        # Group agents by shape (type)
        agents_by_shape = {}
        for agent, info in infos.items():
            shape = self.agent_shapes.get(agent.get_type(), "s")
            agents_by_shape.setdefault(shape, []).append((agent, info))

        # Remove old contagious radius circles
        for circ in self.radius_patches:
            circ.remove()
        self.radius_patches.clear()

        # Update or draw each shape group
        for shape, group in agents_by_shape.items():
            positions = np.array([info["position"] for _, info in group])
            colors = [self._get_agent_color(info) for _, info in group]

            if initial:
                sc = self._ax_sim.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    c=colors,
                    s=15,
                    marker=shape,
                    edgecolors="none",
                )
                self.scatter_artists[shape] = sc
            else:
                self.scatter_artists[shape].set_offsets(positions)
                self.scatter_artists[shape].set_color(colors)

            # Draw contagious radius for infected agents
            for (_, info), (x, y) in zip(group, positions):
                if info.get("infected", False):
                    circle = patches.Circle(
                        (x, y),
                        radius=self.env.infection_radius,
                        color="red",
                        alpha=0.05,
                        lw=0,
                    )
                    self._ax_sim.add_patch(circle)
                    self.radius_patches.append(circle)

    def _draw_graph(self, time_step):
        stats = self.env.get_env_stats()
        total_agents = stats["total_agents"]
        self.infected_percentages.append(stats["num_infected"] / total_agents * 100)
        self.dead_percentages.append((total_agents - stats["num_active"]) / total_agents * 100)
        self.line_infected.set_data(range(1, time_step + 1), self.infected_percentages)
        self.line_dead.set_data(range(1, time_step + 1), self.dead_percentages)

        self._ax_graph.set_xlim(0, self.max_num_steps)
        self._ax_graph.set_ylim(0, 100)

    def _update_info(self):
        stats = self.env.get_env_stats()
        total_agents = stats["total_agents"]
        # --- Update Info Box ---
        num_infected = stats["num_infected"]
        num_dead = total_agents - stats["num_active"]
        num_alive = stats["num_active"]
        info_str = (
            f"Infected: {num_infected}\n"
            f"Dead: {num_dead}\n"
            f"Alive: {num_alive}\n"
            f"Total: {total_agents}"
        )
        self.info_text.set_text(info_str)

    def render(self, time_step):
        initial = time_step == 1
        if initial:
            self._init_render()
        else:
            self._draw_agents(initial)
        self._draw_graph(time_step)
        self._update_info()

        self._time_text.set_text(f"t = {time_step}")

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.1)

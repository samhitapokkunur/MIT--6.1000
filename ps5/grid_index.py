from collections import defaultdict

class GridIndex:
    def __init__(self, points, objects, radius):
        """
        Initializes a grid index for spatial hashing for efficient neighbor queries.
        """
        self.radius = radius
        self.grid = defaultdict(list)
        for (x, y), obj in zip(points, objects):
            cell = self._cell_coords(x, y)
            self.grid[cell].append((x, y, obj))

    def _cell_coords(self, x, y):
        """
        Compute the cell coordinates for a given point.
        """
        return (int(x // self.radius), int(y // self.radius))

    def query_radius(self, qx, qy, r):
        """
        Returns a list of (x, y, obj) tuples for all objects within radius r
        of the query point (qx, qy).
        """
        cx, cy = self._cell_coords(qx, qy)
        cells_to_check = [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        r2 = r * r
        neighbors = []
        for cell in cells_to_check:
            for x, y, obj in self.grid.get(cell, []):
                if (x - qx) ** 2 + (y - qy) ** 2 <= r2:
                    neighbors.append((x, y, obj))
        return neighbors
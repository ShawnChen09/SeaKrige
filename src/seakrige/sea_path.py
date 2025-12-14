import heapq

import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from shapely.geometry import LineString
from shapely.ops import unary_union

from .config import Config


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, u, v, weight=1.0):
        if u not in self.edges:
            self.edges[u] = {}
        if v not in self.edges:
            self.edges[v] = {}
        self.edges[u][v] = weight
        self.edges[v][u] = weight

    def get_neighbors(self, node_id):
        return self.edges.get(node_id, {}).keys()

    def copy(self):
        new_G = Graph()
        new_G.nodes = self.nodes.copy()
        new_G.edges = {k: v.copy() for k, v in self.edges.items()}
        return new_G

    def _dijkstra(self, start, end):
        distances = {node: float("inf") for node in self.nodes}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                weight = self.edges[current][neighbor]
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        raise ValueError("No path exists")

    def dijkstra(self, start, end):
        self.path = self._dijkstra(start, end)
        self.length = 0
        for i in range(len(self.path) - 1):
            self.length += self.edges[self.path[i]][self.path[i + 1]]


class SeaPath:
    def __init__(self, shapefile, verbose=True):
        self.config = Config("seapath", verbose=verbose)

        self.load_shp(shapefile)
        self.extract_v()
        self.build_graph()
        self._path_cache = {}

    def load_shp(self, shapefile):
        self.gdf = pyogrio.read_dataframe(shapefile)
        self.crs = self.gdf.crs
        self.obs = unary_union(self.gdf.geometry)
        self.is_multipolygon = hasattr(self.obs, "geoms")

        if hasattr(self.obs, "bounds"):
            self.bounds = self.obs.bounds
        else:
            self.bounds = (0, 0, 1, 1)

    def extract_v(self):
        verts = []

        if self.is_multipolygon:
            for geom in self.obs.geoms:
                verts.extend(list(geom.exterior.coords[:-1]))
                for interior in geom.interiors:
                    verts.extend(list(interior.coords[:-1]))
        else:
            verts.extend(list(self.obs.exterior.coords[:-1]))
            for interior in self.obs.interiors:
                verts.extend(list(interior.coords[:-1]))

        self.vertices = np.array(verts)

    def is_visible(self, p1, p2):
        l = LineString([p1, p2])
        return not self.obs.crosses(l) and not self.obs.contains(l)

    def build_graph(self):
        self.G = Graph()
        n_vertices = len(self.vertices)

        self.config.logger.info(f"Building graph with {n_vertices} vertices")

        for i, vertex in enumerate(self.vertices):
            self.G.add_node(i, pos=vertex)

        edges_added = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if self.is_visible(self.vertices[i], self.vertices[j]):
                    dist = np.linalg.norm(self.vertices[i] - self.vertices[j])
                    self.G.add_edge(i, j, weight=dist)
                    edges_added += 1

        self.config.logger.info(f"Added {edges_added} edges to graph")

    def calc_direct_distance(self, s: tuple[float, float], t: tuple[float, float]):
        return np.linalg.norm(np.array(s) - np.array(t))

    def calc_path_from_G(self, s: tuple[float, float], t: tuple[float, float]):
        self.config.logger.info(f"Calculating path from {s} to {t}")

        cache_key = (tuple(s), tuple(t))
        if self.config.ENABLE_CACHING and cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            self.path_coords = cached["path_coords"]
            self.path_length = cached["length"]
            self.config.logger.info(
                f"Using cached path, length: {self.path_length:.4f}"
            )
            return self.path_length

        if self.is_visible(s, t):
            direct_dist = self.calc_direct_distance(s, t)
            self.config.logger.info(f"Direct distance: {direct_dist:.4f} units")
            self.path_coords = [s, t]
            self.path_length = direct_dist
            self._path_cache[cache_key] = {
                "path_coords": self.path_coords,
                "length": self.path_length,
            }
            return self.path_length
        else:
            self.config.logger.info("Direct path blocked by land")

        G_temp = self.G.copy()
        source_id = self.add_temp_point(G_temp, s)
        target_id = self.add_temp_point(G_temp, t)

        if self.is_visible(s, t):
            dist = self.calc_direct_distance(s, t)
            G_temp.add_edge(source_id, target_id, weight=dist)

        try:
            G_temp.dijkstra(source_id, target_id)
            self.path_coords = [G_temp.nodes[node]["pos"] for node in G_temp.path]
            self.path_length = G_temp.length

            self._path_cache[cache_key] = {
                "path_coords": self.path_coords,
                "length": self.path_length,
            }
            self.config.logger.info(f"Sea Path length: {self.path_length:.4f} units")
            return self.path_length

        except ValueError:
            raise ValueError("No path exists between coordinates")

    def add_temp_point(self, graph, p):
        point_id = max(graph.nodes.keys()) + 1 if graph.nodes else len(self.vertices)
        graph.add_node(point_id, pos=p)

        for node_id in graph.nodes:
            if node_id == point_id:
                continue
            vertex = graph.nodes[node_id]["pos"]
            if self.is_visible(p, vertex):
                dist = np.linalg.norm(np.array(p) - np.array(vertex))
                graph.add_edge(point_id, node_id, weight=dist)

        return point_id

    def plot_path(self):
        if not hasattr(self, "path_coords"):
            raise ValueError("No path calculated. Call calc_path_from_G() first.")

        path_coords = np.array(self.path_coords)
        plt.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            color=self.config.PATH_COLOR,
            linewidth=self.config.PATH_WIDTH,
            marker="o",
            markersize=4,
        )
        plt.scatter(
            path_coords[0, 0],
            path_coords[0, 1],
            s=100,
            label="Start",
        )
        plt.scatter(
            path_coords[-1, 0],
            path_coords[-1, 1],
            s=100,
            label="End",
        )

    def plot_G(self):
        for node_id, attrs in self.G.nodes.items():
            pos = attrs["pos"]
            plt.scatter(pos[0], pos[1], s=self.config.NODE_SIZE, c="blue", alpha=0.6)

        for node_id, neighbors in self.G.edges.items():
            pos1 = self.G.nodes[node_id]["pos"]
            for neighbor_id in neighbors:
                pos2 = self.G.nodes[neighbor_id]["pos"]
                plt.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    "gray",
                    linewidth=self.config.EDGE_SIZE,
                    alpha=0.6,
                )

    def plot_geomap(self):
        if self.is_multipolygon:
            for geom in self.obs.geoms:
                x, y = geom.exterior.xy
                plt.fill(x, y, facecolor=self.config.LAND_COLOR, alpha=0.7)
        else:
            x, y = self.obs.exterior.xy
            plt.fill(x, y, facecolor=self.config.LAND_COLOR, alpha=0.7)

        plt.gca().set_facecolor(self.config.SEA_COLOR)

    def plot(self, show=True):
        self.plot_geomap()
        self.plot_G()
        self.plot_path()
        if show:
            plt.show()

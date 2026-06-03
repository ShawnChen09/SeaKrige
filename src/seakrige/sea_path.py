import heapq

import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from pyproj import Geod
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

    def _dijkstra(self, start, end, return_path=True):
        if return_path:
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
                    return path[::-1], current_dist

                for neighbor in self.get_neighbors(current):
                    if neighbor in visited:
                        continue

                    weight = self.edges[current][neighbor]
                    distance = current_dist + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (distance, neighbor))
        else:
            distances = {node: float("inf") for node in self.nodes}
            distances[start] = 0
            pq = [(0, start)]
            visited = set()

            while pq:
                current_dist, current = heapq.heappop(pq)

                if current in visited:
                    continue
                visited.add(current)

                if current == end:
                    return current_dist

                for neighbor in self.get_neighbors(current):
                    if neighbor in visited:
                        continue

                    weight = self.edges[current][neighbor]
                    distance = current_dist + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))

        raise ValueError("No path exists")

    def dijkstra(self, start, end, return_path=True):
        if return_path:
            self.path, self.length = self._dijkstra(start, end, True)
            return self.path
        else:
            return self._dijkstra(start, end, False)


class SeaPath:
    def __init__(self, shapefile, verbose=True):
        self.config = Config("seapath", verbose=verbose)
        self.geod = Geod(ellps="WGS84")

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

    def is_visible(self, coord_1, coord_2):
        direct_path = LineString([coord_1, coord_2])
        return not self.obs.crosses(direct_path) and not self.obs.contains(direct_path)

    def is_visible_batch(self, coord_list_1, coord_list_2):
        n_A, n_B = len(coord_list_1), len(coord_list_2)
        visibility_matrix = np.ones((n_A, n_B), dtype=bool)

        for i in range(n_A):
            for j in range(n_B):
                visible = self.is_visible(coord_list_1[i], coord_list_2[j])
                visibility_matrix[i, j] = visible

        return visibility_matrix

    def distance_meter(self, coord_1, coord_2):
        _, _, dist = self.geod.inv(
            coord_1[0],
            coord_1[1],
            coord_2[0],
            coord_2[1],
        )
        return dist

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
                    dist = self.distance_meter(self.vertices[i], self.vertices[j])
                    self.G.add_edge(i, j, weight=dist)
                    edges_added += 1

        self.config.logger.info(f"Added {edges_added} edges to graph")

    def calc_direct_distance(
        self, coord_1: tuple[float, float], coord_2: tuple[float, float]
    ):
        return self.distance_meter(coord_1, coord_2)

    def add_temp_point(self, graph, point):
        point_id = max(graph.nodes.keys()) + 1 if graph.nodes else len(self.vertices)
        graph.add_node(point_id, pos=point)

        for node_id in graph.nodes:
            if node_id == point_id:
                continue
            vertex = graph.nodes[node_id]["pos"]
            if self.is_visible(point, vertex):
                dist = self.distance_meter(point, vertex)
                graph.add_edge(point_id, node_id, weight=dist)

        return point_id

    def calc_path(
        self,
        coord_1: tuple[float, float],
        coord_2: tuple[float, float],
        check_visibility=True,
    ):
        cache_key = (tuple(coord_1), tuple(coord_2))
        if self.config.ENABLE_CACHING and cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            self.path_length = cached["length"]
            return self.path_length

        if check_visibility and self.is_visible(coord_1, coord_2):
            direct_dist = self.calc_direct_distance(coord_1, coord_2)
            self.path_coords = [coord_1, coord_2]
            self.path_length = direct_dist
            self._path_cache[cache_key] = {
                "path_coords": self.path_coords,
                "length": self.path_length,
            }
            return self.path_length

        try:
            G_temp = self.G.copy()
            temp_point_1 = self.add_temp_point(G_temp, coord_1)
            temp_point_2 = self.add_temp_point(G_temp, coord_2)

            path_length = G_temp.dijkstra(temp_point_1, temp_point_2, False)

            self._path_cache[cache_key] = {
                "length": path_length,
            }

            self._last_calc = {
                "coord_1": coord_1,
                "coord_2": coord_2,
            }

            return path_length

        except ValueError:
            raise ValueError("No path exists between coordinates")

    def calc_multiple_paths_batch(self, point_pairs):
        results = []

        for i, (coord_1, coord_2) in enumerate(point_pairs):
            try:
                results.append(self.calc_path(coord_1, coord_2, check_visibility=False))
            except ValueError:
                results.append(self.distance_meter(coord_1, coord_2) * 1.2)
            except KeyboardInterrupt:
                self.config.logger.info(
                    f"Interrupted at {i}/{len(point_pairs)} calculations"
                )
                raise

        return results

    def plot_path(self):
        if not hasattr(self, "path_coords"):
            if not hasattr(self, "_last_calc"):
                raise ValueError("No path calculated. Call calc_path_from_G() first.")

            # Recalculate path for plotting
            last_calc = self._last_calc
            coord_1, coord_2 = last_calc["coord_1"], last_calc["coord_2"]

            G_temp = self.G.copy()
            temp_point_1 = self.add_temp_point(G_temp, coord_1)
            temp_point_2 = self.add_temp_point(G_temp, coord_2)

            path_indices = G_temp.dijkstra(temp_point_1, temp_point_2, True)
            self.path_coords = [G_temp.nodes[idx]["pos"] for idx in path_indices]

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
            label="coord_1",
        )
        plt.scatter(
            path_coords[-1, 0],
            path_coords[-1, 1],
            s=100,
            label="coord_2",
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

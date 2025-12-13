import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pykrige.core
import pykrige.ok
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point

from .config import Config
from .sea_path import SeaPath


class SeaKrige:
    def __init__(
        self,
        longitude,
        latitude,
        z_values,
        shapefile,
        variogram_model="spherical",
        variogram_parameters=[1.0, 0.5, 0.1],
        verbose=True,
    ):
        self.config = Config(verbose=verbose)

        self.gdf = gpd.read_file(shapefile)
        self.sea_path = SeaPath(shapefile, verbose=False)

        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)
        z_values = np.asarray(z_values)

        self.config.logger.info(f"Validating {len(longitude)} data points")
        self.validate_sea_points(longitude, latitude)

        self.longitude = longitude
        self.latitude = latitude
        self.z_values = z_values

        self.config.logger.info(
            f"Creating OrdinaryKriging with {variogram_model} model"
        )

        self.OK = OrdinaryKriging(
            longitude,
            latitude,
            z_values,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            verbose=False,
            enable_plotting=False,
        )

    def monkey_patch_dist_func(self):
        self._original_pdist = pykrige.core.pdist
        self._original_cdist = pykrige.ok.cdist
        self._original_core_cdist = pykrige.core.cdist

        pykrige.core.pdist = self.sea_path_pdist
        pykrige.core.cdist = self.sea_path_cdist
        pykrige.ok.cdist = self.sea_path_cdist

    def restore_dist_func(self):
        pykrige.core.pdist = self._original_pdist
        pykrige.core.cdist = self._original_cdist
        pykrige.ok.cdist = self._original_core_cdist

    def validate_sea_points(self, longitude, latitude):
        land_geometry = self.gdf.geometry.union_all()
        points_inside = []
        points_outside = []

        for i, (lon, lat) in enumerate(zip(longitude, latitude)):
            point = Point(lon, lat)
            if land_geometry.contains(point):
                points_inside.append(i)
            else:
                points_outside.append(i)

        if points_inside and points_outside:
            raise ValueError(
                f"Mixed point locations: {len(points_inside)} inside polygon, "
                f"{len(points_outside)} outside polygon. All points must be either "
                "inside or outside the polygon."
            )

    def sea_path_pdist(self, X, metric="euclidean"):
        if metric != "euclidean":
            return self._original_pdist(X, metric)

        n = len(X)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    dist = self.sea_path.calc_path_from_G(X[i], X[j])
                except ValueError:
                    dist = np.sqrt((X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2)
                distances.append(dist)
        return np.array(distances)

    def sea_path_cdist(self, XA, XB, metric="euclidean"):
        if metric != "euclidean":
            return self._original_cdist(XA, XB, metric)

        distances = np.zeros((len(XA), len(XB)))
        for i, (xa, ya) in enumerate(XA):
            for j, (xb, yb) in enumerate(XB):
                try:
                    dist = self.sea_path.calc_path_from_G([xa, ya], [xb, yb])
                except ValueError:
                    dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
                distances[i, j] = dist
        return distances

    def create_land_mask(self, gridx, gridy):
        land_geometry = self.gdf.geometry.union_all()
        self.mask = np.zeros((len(gridy), len(gridx)), dtype=bool)

        for i, y in enumerate(gridy):
            for j, x in enumerate(gridx):
                point = Point(x, y)
                self.mask[i, j] = land_geometry.contains(point)

    def execute(self, gridx, gridy):
        self.config.logger.info(f"Executing kriging on {len(gridx)}x{len(gridy)} grid")

        self.monkey_patch_dist_func()
        self.create_land_mask(gridx, gridy)

        if self.config.verbose:
            land_points = np.sum(self.mask)
            total_points = self.mask.size
            self.config.logger.info(f"Masked {land_points}/{total_points} land points")

        z, ss = self.OK.execute("masked", gridx, gridy, mask=self.mask)
        self.restore_dist_func()

        self.config.logger.info("Kriging execution completed")

        return z, ss

    def plot_results(self, z, gridx, gridy, show=True, add_scatter=True):
        fig, ax = plt.subplots()

        self.gdf.plot(ax=ax, facecolor=self.config.LAND_COLOR, alpha=0.7)

        im = ax.imshow(
            z,
            extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()],
            origin="lower",
            cmap="viridis",
            alpha=0.8,
        )

        if add_scatter:
            ax.scatter(
                self.longitude, self.latitude, c="red", s=50, marker="x", zorder=10
            )

        ax.set_xlim(gridx.min(), gridx.max())
        ax.set_ylim(gridy.min(), gridy.max())
        plt.colorbar(im, ax=ax, shrink=0.8)

        if show:
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax

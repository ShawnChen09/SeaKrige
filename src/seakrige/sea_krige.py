import time

import matplotlib.pyplot as plt
import numpy as np
import pykrige
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
        self.config = Config("seakrige", verbose=verbose)

        self.config.logger.info(f"Loading shapefile: {shapefile}")
        self.sea_path = SeaPath(shapefile, verbose=False)

        SeaKrige.validate_data_points(longitude, latitude, z_values)
        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)
        z_values = np.asarray(z_values)

        self.config.logger.info(f"Loading {len(longitude)} data points")
        self.longitude = longitude
        self.latitude = latitude
        self.z_values = z_values

        self.variogram_model = variogram_model
        self.variogram_parameters = variogram_parameters

    def validate_data_points(longitude, latitude, z_values):
        if len(longitude) != len(latitude) or len(longitude) != len(z_values):
            raise ValueError(
                "longitude, latitude and z_values must have the same length"
            )
        if -180 > np.min(longitude) or np.max(longitude) > 180:
            raise ValueError("longitude must be between -180 and 180")
        if -90 > np.min(latitude) or np.max(latitude) > 90:
            raise ValueError("latitude must be between -90 and 90")

    def get_ok(self):
        self.config.logger.info(
            f"Creating OrdinaryKriging with '{self.variogram_model}' model"
        )
        self.config.logger.info(f"Variogram parameters: {self.variogram_parameters}")
        self.OK = OrdinaryKriging(
            self.longitude,
            self.latitude,
            self.z_values,
            variogram_model=self.variogram_model,
            variogram_parameters=self.variogram_parameters,
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

    def sea_path_pdist(self, X, metric="euclidean"):
        if metric != "euclidean":
            return self._original_pdist(X, metric)

        n = len(X)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    dist = self.sea_path.calc_path(X[i], X[j])
                except ValueError:
                    dist = np.sqrt((X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2)
                distances.append(dist)
        return np.array(distances)

    def sea_path_cdist(self, XA, XB, metric="euclidean"):
        if metric != "euclidean":
            return self._original_cdist(XA, XB, metric)

        start_time = time.time()
        euclidean_matrix = self._original_cdist(XA, XB, "euclidean")

        if self.config.verbose:
            self.config.logger.info("Checking visibility for point pairs")

        visibility_matrix = self.sea_path.is_visible_batch(XA, XB)
        blocked_pairs = np.where(~visibility_matrix)
        sea_distances = euclidean_matrix.copy()
        total_blocked = len(blocked_pairs[0])

        self.config.logger.info(
            f"Calculating sea-path for {total_blocked}/{euclidean_matrix.size} blocked point pairs"
        )

        if total_blocked > 0:
            blocked_point_pairs = [(XA[i], XB[j]) for i, j in zip(*blocked_pairs)]

            try:
                batch_distances = self.sea_path.calc_multiple_paths_batch(
                    blocked_point_pairs
                )

                for idx, (i, j) in enumerate(zip(*blocked_pairs)):
                    sea_distances[i, j] = batch_distances[idx]

            except (AttributeError, Exception):
                self.config.logger.warning(
                    "Batch calculation failed. Falling back to individual calculations."
                )
                for idx, (i, j) in enumerate(zip(*blocked_pairs)):
                    if self.config.verbose and idx % 1000 == 0:
                        self.config.logger.info(
                            f"Progress: {idx}/{total_blocked} ({100 * idx / total_blocked:.1f}%)"
                        )

                    try:
                        sea_distances[i, j] = self.sea_path.calc_path(
                            XA[i], XB[j], check_visibility=False
                        )
                    except ValueError:
                        pass
        self.config.logger.info(
            f"Sea-path calculation completed in {time.time() - start_time:.2f} seconds"
        )
        return sea_distances

    def create_land_mask(self, gridx, gridy):
        land_geometry = self.sea_path.gdf.geometry.union_all()
        self.mask = np.zeros((len(gridy), len(gridx)), dtype=bool)

        for i, y in enumerate(gridy):
            for j, x in enumerate(gridx):
                point = Point(x, y)
                self.mask[i, j] = land_geometry.contains(point)

    def execute(self, gridx, gridy):
        self.config.logger.info(f"Loading {len(gridx)}x{len(gridy)} grid")
        self.gridx = gridx
        self.gridy = gridy

        if not hasattr(self, "OK"):
            self.get_ok()

        self.monkey_patch_dist_func()

        self.config.logger.info("Creating land mask")
        self.create_land_mask(gridx, gridy)

        if self.config.verbose:
            land_points = np.sum(self.mask)
            total_points = self.mask.size
            self.config.logger.info(f"Masked {land_points}/{total_points} land points")

        self.config.logger.info("Executing kriging ...")
        self.z, self.ss = self.OK.execute("masked", gridx, gridy, mask=self.mask)
        self.config.logger.info("Kriging execution completed")

        self.restore_dist_func()

    def plot(
        self,
        show=True,
        use_alpha=True,
        alpha_resolution=0.05,
        cmap="viridis",
        add_scatter=True,
        scatter_color="red",
        scatter_size=50,
        scatter_marker="x",
    ):
        if not hasattr(self, "z") or not hasattr(self, "ss"):
            self.config.logger.error(
                "No kriging results to plot. Run the `execute` function first."
            )
            return
        if use_alpha:
            alpha = self.calc_alpha(alpha_resolution)
        else:
            alpha = None

        fig, ax = plt.subplots()

        self.sea_path.gdf.plot(ax=ax, facecolor=self.config.LAND_COLOR, alpha=0.7)

        im = ax.imshow(
            self.z,
            extent=[
                self.gridx.min(),
                self.gridx.max(),
                self.gridy.min(),
                self.gridy.max(),
            ],
            origin="lower",
            cmap=cmap,
            alpha=alpha,
        )

        if add_scatter:
            ax.scatter(
                self.longitude,
                self.latitude,
                c=scatter_color,
                s=scatter_size,
                marker=scatter_marker,
                zorder=10,
            )

        ax.set_xlim(self.gridx.min(), self.gridx.max())
        ax.set_ylim(self.gridy.min(), self.gridy.max())
        plt.colorbar(im, ax=ax, shrink=0.8)

        if show:
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax

    def calc_alpha(self, res):
        valid_variance = self.ss[~np.isnan(self.ss)]
        min_var, max_var = valid_variance.min(), valid_variance.max()

        normalized_variance = (self.ss - min_var) / (max_var - min_var)

        variance_alpha = 1.0 - normalized_variance
        variance_alpha = np.ceil(variance_alpha / res) * res
        return np.clip(variance_alpha, 0, 1.0)

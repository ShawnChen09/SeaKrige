import os
import tempfile
from contextlib import contextmanager

import geopandas as gpd
import matplotlib.pyplot as plt
from seakrige.sea_path import SeaPath
from shapely.geometry import Polygon


@contextmanager
def temp_shp():
    with tempfile.TemporaryDirectory() as temp_dir:
        coords = [
            (120.195, 24.20),
            (120.205, 24.20),
            (120.205, 24.05),
            (120.195, 24.05),
        ]
        ploygon = Polygon(coords)

        shp_path = os.path.join(temp_dir, "temp.shp")

        gdf = gpd.GeoDataFrame({"name": ["peninsula"]}, geometry=[ploygon])
        gdf.crs = "EPSG:4326"
        gdf.to_file(shp_path)

        yield shp_path


def main():
    coord_1 = (120.19, 24.12)
    coord_2 = (120.21, 24.08)
    print(f"Test Route: {coord_1} -> {coord_2}")

    with temp_shp() as shapefile_path:
        seapath = SeaPath(shapefile=shapefile_path)

    path_length = seapath.calc_path(coord_1=coord_1, coord_2=coord_2)

    seapath.plot(show=False)

    plt.xlim(120.175, 120.225)
    plt.ylim(24.00, 24.25)
    plt.title("Sea Path")
    plt.annotate(
        f"Length: {path_length / 1000:.2f} km",
        xy=(0.14, 0.83),
        xycoords="figure fraction",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    print("completed!")

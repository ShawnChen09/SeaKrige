import matplotlib.pyplot as plt
import numpy as np
from example_SeaPath import temp_shp
from seakrige.sea_krige import SeaKrige


def main():
    longitude = np.array([120.19, 120.21, 120.18, 120.206, 120.19])
    latitude = np.array([24.22, 24.03, 24.125, 24.125, 24.04])
    z_values = np.array([10.0, 15.0, 15.0, -100.0, 11.0])
    print(f"Known points: {len(longitude)}")

    gridx = np.linspace(120.17, 120.23, 101)
    gridy = np.linspace(24.0, 24.25, 121)

    with temp_shp() as shapefile_path:
        seakrige = SeaKrige(
            longitude,
            latitude,
            z_values,
            shapefile_path,
            variogram_model="spherical",
            variogram_parameters=[1.0, 0.7, 0.1],
            verbose=True,
        )

    seakrige.execute(gridx, gridy)

    fig, ax = seakrige.plot(show=False, use_alpha=True, alpha_resolution=0.1)

    ax.set_aspect("auto")
    ax.set_title("Sea Kriging")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


if __name__ == "__main__":
    main()

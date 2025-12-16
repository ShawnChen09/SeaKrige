# SeaKrige

A kriging implementation that uses obstacle-avoiding path distances instead of Euclidean distances for spatial interpolation. Accounts for spatial barriers defined by polygon geometries.

## Demo

| SeaPath | SeaKrige |
|---------------------------------------------|-------------------------------------------|
| <img src="https://github.com/ShawnChen09/SeaKrige/raw/main/img/sea_path.jpg" width="300"/> | <img src="https://github.com/ShawnChen09/SeaKrige/raw/main/img/sea_krige.jpg" width="300"/> |

## Quick Start

### Installation

1. Install dependencies (pyKrige, geopandas, matplotlib):
```sh
python -m pip install -r requirements.txt
```

2. Install the package:
```sh
python -m pip install -e .
```

### Basic Usage
See the `example/` folder for usage examples demonstrating both `SeaPath` and `SeaKrige` functionality.

## Modules

### `SeaPath`
Uses Dijkstra to calculate shortest obstacle-free distances between coordinates.

### `SeaKrige`
Performs ordinary kriging interpolation using SeaPath distances instead of Euclidean distances. Integrates with [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/) through monkey-patching of distance functions.

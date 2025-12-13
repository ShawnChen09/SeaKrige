# SeaKrige

A barrier-aware kriging implementation that uses obstacle-avoiding path distances instead of Euclidean distances for spatial interpolation. Accounts for any type of spatial barriers defined by polygon geometries.

## Demo

| SeaPath | SeaKrige |
|---------------------------------------------|-------------------------------------------|
| <img src="https://github.com/ShawnChen09/seakrige/raw/main/img/sea_path.jpg" width="300"/> | <img src="https://github.com/ShawnChen09/seakrige/raw/main/img/sea_krige.jpg" width="300"/> |

## Installation

1. Install required packages:
```sh
python -m pip install -r requirements.txt
```

2. Install the package:
```sh
python -m pip install -e .
```

## Modules

### `SeaPath`
Calculates shortest obstacle-free distances between coordinates while avoiding barriers defined in shapefiles. Uses visibility graph algorithms to find optimal paths through accessible areas.

### `SeaKrige`
Performs ordinary kriging interpolation using barrier-aware path distances instead of Euclidean distances. Integrates with [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/) through monkey-patching of distance functions.

### `Config`
Manages configuration settings for both modules, including customizable parameters for boundary margins, visualization settings, etc.



## Usage
See the `example/` folder for complete usage examples and test scripts demonstrating both `SeaPath` and `SeaKrige` functionality.
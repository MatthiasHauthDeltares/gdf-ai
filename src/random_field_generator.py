import pickle
import pandas as pd
import gstools as gs
import numpy as np


def trend(x, y):
    """Increasing trend of the mean in y direction"""
    return 0.02 * y ** 2


def get_field(x_max: float, y_max: float, x_resolution: float, y_resolution: float, model_type: str,
              from_pickle: bool) -> tuple[np.array, np.array, np.array]:
    """Generate a random field using gstools for a bounding box of x_max and y_max with a resolution of x_resolution
    The point (0, 0) is the upper left corner of the field, the point (x_max, y_max) is the lower right corner of the
    field.

    :param x_max: maximum x value
    :param y_max: maximum y value
    :param x_resolution: resolution in x direction
    :param y_resolution: resolution in y direction
    :param from_pickle: if True, the field is loaded from a pickle file, otherwise it is generated

    :return: the field as a 2D numpy array and the x and y coordinates as 1D numpy arrays

    """
    x = np.arange(0, x_max + x_resolution, x_resolution)  # 100m in x-direction
    y = np.arange(0, y_max + y_resolution, y_resolution)  # 20m depth

    if from_pickle:
        with open('field.pkl', 'rb') as f:
            field = pickle.load(f)
        return field, x, y

    theta_v = 2
    theta_h = 50
    qt_mean = 8
    qt_vart = 4

    if model_type == 'spherical':
        model = gs.Spherical(dim=2, var=qt_vart, len_scale=[theta_h, theta_v])
    elif model_type == 'gaussian':
        model = gs.Gaussian(dim=2, var=qt_vart, len_scale=[theta_h, theta_v])
    else:
        raise NotImplementedError("Model not implemented")

    srf = gs.SRF(model, mean=qt_mean, trend=trend)
    field = srf.structured([x, y])

    # save field in pickle file
    with open('field.pkl', 'wb') as f:
        pickle.dump(field, f)

    return field.T, x, y


def get_field_dataframe(field: np.array, x_values: np.array, y_values: np.array) -> pd.DataFrame:
    """Convert a field to a dataframe"""

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Flatten the 2D arrays to 1D
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    data_flat = field.ravel()

    # Create a DataFrame
    df = pd.DataFrame({
        'X': x_flat,
        'Y': y_flat,
        'value': data_flat
    })
    return df


def get_gdf_dataframe(field: np.array, x_values: np.array, y_values: np.array) -> pd.DataFrame:

    x_grid, y_grid = np.meshgrid(x_values, y_values)



    for x in x_values:
        for y in y_values:
            pass
    pass


def calc_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the distance between two points"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


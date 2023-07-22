import pickle
import pandas as pd
import gstools as gs
import numpy as np


def trend(x, y):
    """Increasing trend of the mean in y direction"""
    return 0.02 * y ** 2


def get_field(from_pickle=False):
    """Generate a random field"""
    x_resolution = 0.5
    y_resolution = 0.05
    x = np.arange(0, 100.5, x_resolution)  # 100m in x-direction
    y = np.arange(0, 20.05, y_resolution)  # 20m depth

    if from_pickle:
        with open('field.pkl', 'rb') as f:
            field = pickle.load(f)
        return field, x, y

    theta_v = 2
    theta_h = 50
    qt_mean = 8
    qt_vart = 4

    # model = gs.Gaussian(dim=2, var=qt_vart, len_scale=[theta_h, theta_v])
    model = gs.Spherical(dim=2, var=qt_vart, len_scale=[theta_h, theta_v])

    srf = gs.SRF(model, mean=qt_mean, trend=trend)
    field = srf.structured([x, y])

    # save field in pickle file
    with open('field.pkl', 'wb') as f:
        pickle.dump(field, f)

    return field, x, y


def get_field_dataframe(field: np.array, x: np.array, y: np.array) -> pd.DataFrame:
    """Convert a field to a dataframe"""

    x_grid, y_grid = np.meshgrid(x, y)

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

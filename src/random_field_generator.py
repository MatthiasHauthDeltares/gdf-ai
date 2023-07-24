import pickle
import pandas as pd
import gstools as gs
import numpy as np


class Field2D():

    def __init__(self, field_data: np.array, name: str):
        self.field_data = field_data
        self.name = name


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
    field_data = srf.structured([x, y])
    field = Field2D(field_data.T, 'base')

    # save field in pickle file
    with open('field.pkl', 'wb') as f:
        pickle.dump(field, f)

    return field, x, y


def get_field_dataframe(field: Field2D, x_values: np.array, y_values: np.array) -> tuple[pd.DataFrame, list[str]]:
    """Convert a field to a dataframe and return the feature names"""

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Flatten the 2D arrays to 1D
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    data_flat = field.field_data.ravel()

    # Create a DataFrame
    df = pd.DataFrame({
        'X': x_flat,
        'Y': y_flat,
        'value': data_flat
    })
    return df, ["X", "Y"]


def get_gdf_dataframe(field: np.array, x_values: np.array, y_values: np.array,
                      CPT_x_locations: list[float]) -> tuple[pd.DataFrame, list[str]]:
    """Convert a field to a dataframe with all appended GDFs and return the feature names

    For a 2D field:

    The four corners are defined as:
    A ----------------- B
    |                   |
    |                   |
    D ----------------- C


    :param field: the field as a 2D numpy array
    :param x_values: the x values of the field
    :param y_values: the y values of the field
    :param CPT_x_locations: the x locations of the CPTs

    :return: tuple of the dataframe and the feature names
    """

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    df, _ = get_field_dataframe(field, x_values, y_values)
    # Remove columns "X"
    df = df.drop(columns=["X"])

    # Add testing line distance field:
    for i, x_loc in enumerate(CPT_x_locations):
        df[f"line_{i}"] = np.sqrt((x_grid - x_loc) ** 2).ravel()

    # Add the corner distance fields:
    corners = [(x_values[0], y_values[0]),  # corner A
               (x_values[0], y_values[-1]),  # corner D
               (x_values[-1], y_values[0]),  # corner B
               (x_values[-1], y_values[-1])]  # corner C

    corner_distances = np.empty((4, len(y_values), len(x_values)))
    for i, (x_corner, y_corner) in enumerate(corners):
        corner_distances[i] = np.sqrt((x_grid - x_corner) ** 2 + (y_grid - y_corner) ** 2)

    # Add the corner distance fields to the dataframe:
    df['corner_A'] = corner_distances[0].ravel()
    df['corner_D'] = corner_distances[1].ravel()
    df['corner_B'] = corner_distances[2].ravel()
    df['corner_C'] = corner_distances[3].ravel()

    return df, [col for col in df.columns if col not in ['value', 'CPT_id']]



def calc_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the distance between two points"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

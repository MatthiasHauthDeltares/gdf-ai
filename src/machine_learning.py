import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def split_field_train_test(field_df: pd.DataFrame,
                           number_CPT: int,
                           x_values: np.array,
                           split_cpt: bool,
                           features: list[str]
                           ):
    """Split a field into train and test data
    Right now all the CPT voxels from the field are considered as 'training' data and we test on the other voxels of
    the field where CPT are not available: this would not be applicable in practice because such field would not be
    available since we only have access to the CPT data.

    :param field_df: dataframe containing the field data
    :param number_CPT: number of CPT to use for training
    :param x_values: x values of the field
    :param split_cpt: if True, the CPT data are split into train and test data
    :param features: list of features names to use
    """

    if split_cpt:
        raise NotImplementedError()
    # Selecting values at these indices

    # Prepare Data
    data = field_df
    X = data[features].values
    Y = data['value'].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.01)

    return x_train, x_test, y_train, y_test


def train_model(regressor_type: str, x_train, y_train):
    if regressor_type == 'RandomForestRegressor':
        regressor = RandomForestRegressor(n_estimators=50)
    elif regressor_type == 'ExtraTreesRegressor':
        regressor = ExtraTreesRegressor()
    elif regressor_type == 'MLPRegressor':
        regressor = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    elif regressor_type == 'GradientBoostingRegressor':
        regressor = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)
    else:
        raise NotImplementedError()

    regressor.fit(x_train, y_train)

    return regressor

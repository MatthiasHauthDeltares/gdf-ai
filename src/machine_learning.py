import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.random_field_generator import get_field_dataframe, get_gdf_dataframe


def split_field_train_test(field_df: pd.DataFrame,
                           number_CPT: int,
                           x_CPT: np.array,
                           split_cpt: bool,
                           features: list[str]
                           ):
    """Split a field into train and test data
    Right now all the CPT voxels from the field are considered as 'training' data and we test on the other voxels of
    the field where CPT are not available: this would not be applicable in practice because such field would not be
    available since we only have access to the CPT data.

    :param field_df: dataframe containing the field data
    :param number_CPT: number of CPT to use for training
    :param x_CPT: x coordinates of the CPT where the values of the field are known
    :param split_cpt: if True, the CPT data are split into train and test data
    :param features: list of features names to use
    """

    if split_cpt:
        raise NotImplementedError()

    data = field_df[field_df['X'].isin(x_CPT)].copy()

    X = data[features].values
    Y = data['value'].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.001)

    return x_train, x_test, y_train, y_test


def train_model(regressor_type: str, x_train, y_train, ml_params: dict):
    if regressor_type == 'random_forest':
        regressor = RandomForestRegressor(n_estimators=ml_params['number_of_trees'], criterion='absolute_error')
    elif regressor_type == 'extra_trees':
        regressor = ExtraTreesRegressor(n_estimators=ml_params['number_of_trees'])
    elif regressor_type == 'MLPRegressor':
        regressor = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    elif regressor_type == 'GradientBoostingRegressor':
        regressor = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)
    else:
        raise NotImplementedError()

    regressor.fit(x_train, y_train)

    return regressor



def run_ml_flow(x_values, number_CPT, split_cpt, feature_type, regressor, ml_params, field, y_values):
    x_CPTs = x_values[np.linspace(0, len(x_values) - 1, number_CPT).round().astype(int)]

    if feature_type == 'cartesian':
        field_data, features = get_field_dataframe(field, x_values, y_values)
    elif feature_type == 'gdf':
        field_data, features = get_gdf_dataframe(field, x_values, y_values, x_CPTs)
    else:
        raise NotImplementedError()

    # Selecting values at these indices
    x_train, x_test, y_train, y_test = split_field_train_test(field_data,
                                                              number_CPT,
                                                              x_CPTs,
                                                              split_cpt=split_cpt,
                                                              features=features)

    trained_regressor = train_model(regressor, x_train, y_train, ml_params)

    X_field = field_data[features].values
    y_field_predicted = trained_regressor.predict(X_field)

    return y_field_predicted
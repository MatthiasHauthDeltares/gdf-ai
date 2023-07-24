import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def split_field_train_test(field_df: pd.DataFrame,
                           number_CPT: int,
                           x_values: np.array,
                           split_cpt: bool):
    """Split a field into train and test data
    Right now all the CPT voxels from the field are considered as 'training' data and we test on the other voxels of
    the field where CPT are not available: this would not be applicable in practice because such field would not be
    available since we only have access to the CPT data.
    """

    if split_cpt:
        raise NotImplementedError()
    # Selecting values at these indices
    x_CPT = x_values[np.linspace(0, len(x_values) - 1, number_CPT).round().astype(int)]
    CPT_id = [i for i in range(number_CPT)]
    # Prepare Data
    data = field_df[field_df['X'].isin(x_CPT)].copy()

    cpt_dict = dict(zip(x_CPT, CPT_id))
    data.loc[:, 'CPT_id'] = data['X'].map(cpt_dict)

    X = data[['X', 'Y']].values
    Y = data['value'].values
    print(data)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.01)

    return x_train, x_test, y_train, y_test


def train_model(regressor_type: str, x_train, y_train):
    if regressor_type == 'RandomForestRegressor':
        regressor = RandomForestRegressor(n_estimators=5)
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



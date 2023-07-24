import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.machine_learning import split_field_train_test, train_model
from src.plot_field import plot_field, plot_field_comparison
from src.random_field_generator import get_field, get_field_dataframe, get_gdf_dataframe, Field2D


feature_type = "gdf"  # gdf or cartesian
split_cpt = False
number_CPT = 5
regressor = "RandomForestRegressor" #or 'RandomForestRegressor' or 'GradientBoostingRegressor'


field, x_values, y_values = get_field(x_max=100, y_max=20, x_resolution=0.5, y_resolution=0.05, model_type='spherical',
                        from_pickle=True)


if feature_type == 'cartesian':
    field_data, features = get_field_dataframe(field, x_values, y_values)
elif feature_type == 'gdf':
    x_CPTs = x_values[np.linspace(0, len(x_values) - 1, number_CPT).round().astype(int)]
    field_data, features = get_gdf_dataframe(field, x_values, y_values, x_CPTs)
else:
    raise NotImplementedError()


# Selecting values at these indices
x_train, x_test, y_train, y_test = split_field_train_test(field_data,
                                                          number_CPT,
                                                          x_values,
                                                          split_cpt=split_cpt,
                                                          features=features)


trained_regressor = train_model(regressor, x_train, y_train)

# y_pred = trained_regressor.predict(x_test)

#
# plt.figure()
# plt.scatter(y_test, y_pred)
# plt.show()


X_field = field_data[features].values
y_field_true = field_data['value'].values

y_field_predicted = trained_regressor.predict(X_field)


field_predicted = Field2D(y_field_predicted.reshape(len(y_values), len(x_values)), "XY RandomForest")
# plot_field(field_predicted.T, x, y)

plot_field_comparison([field, field_predicted], x_values, y_values)

# print(trained_regressor.score(y_field_true, y_field_predicted))

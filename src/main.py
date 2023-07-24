import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.machine_learning import split_field_train_test, train_model
from src.plot_field import plot_field, plot_field_comparison
from src.random_field_generator import get_field, get_field_dataframe, get_gdf_dataframe

field, x, y = get_field(x_max=100, y_max=20, x_resolution=0.5, y_resolution=0.05, model_type='spherical',
                        from_pickle=False)



field_data = get_field_dataframe(field, x, y)

field_data_enriched = get_gdf_dataframe(field, x, y)

feature_type = "gdf"  # gdf or cartesian



# create Dataset for training
number_CPT = 5
# Generating evenly spaced indices

# Selecting values at these indices
x_train, x_test, y_train, y_test = split_field_train_test(field_data,
                                                          number_CPT,
                                                          x,
                                                          split_cpt=False)


trained_regressor = train_model('RandomForestRegressor', x_train, y_train)

# y_pred = trained_regressor.predict(x_test)

#
# plt.figure()
# plt.scatter(y_test, y_pred)
# plt.show()


X_field = field_data[['X', 'Y']].values
y_field_true = field_data['value'].values

y_field_predicted = trained_regressor.predict(X_field)

print(y_field_predicted.shape)

field_predicted = y_field_predicted.reshape(len(y), len(x))
# plot_field(field_predicted.T, x, y)

plot_field_comparison([field, field_predicted], x, y)

print(trained_regressor.score(y_field_true, y_field_predicted))

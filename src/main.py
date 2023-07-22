import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.machine_learning import split_field_train_test
from src.plot_field import plot_field, plot_field_comparison
from src.random_field_generator import get_field, get_field_dataframe

import matplotlib.pyplot as plt

a = RandomForestClassifier

field, x, y = get_field(from_pickle=True)
field_df = get_field_dataframe(field, x, y)

X_field = field_df[['X', 'Y']].values
y_field = field_df['value'].values

# print(field)
# plot_field(field.T, x, y)


# create Dataset for training
number_CPT = 10
# Generating evenly spaced indices

# Selecting values at these indices
x_train, x_test, y_train, y_test = split_field_train_test(field_df,
                                                          number_CPT,
                                                          x,
                                                          split_cpt=False)

# regressor = RandomForestRegressor(n_estimators=5)
# regressor = ExtraTreesRegressor()
regressor = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
# regressor = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#
# plt.figure()
# plt.scatter(y_test, y_pred)
# plt.show()


y_field_predicted = regressor.predict(X_field)

field_predicted = y_field_predicted.reshape(len(x), len(y))
# plot_field(field_predicted.T, x, y)

plot_field_comparison([field.T, field_predicted.T], x, y)


print(regressor.score(y_field, y_field_predicted))

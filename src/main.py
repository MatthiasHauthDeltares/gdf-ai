import numpy as np

from src.machine_learning import split_field_train_test, train_model, run_ml_flow
from src.plot_field import plot_field_comparison
from src.random_field_generator import get_field, get_field_dataframe, get_gdf_dataframe, Field2D

split_cpt = False
number_CPT = 5
regressor = "GradientBoostingRegressor" #or 'RandomForestRegressor' or 'GradientBoostingRegressor'
number_of_trees = 30
ml_params = {"number_of_trees": number_of_trees}


field, x_values, y_values = get_field(x_max=100, y_max=20, x_resolution=0.5, y_resolution=0.05, model_type='spherical',
                        from_pickle=True)

y_predicted_1 = run_ml_flow(number_CPT=number_CPT,
                            split_cpt=split_cpt,
                            feature_type='cartesian',
                            regressor=regressor,
                            ml_params=ml_params,
                            field=field,
                            x_values=x_values,
                            y_values=y_values
                            )

y_predicted_2 = run_ml_flow(number_CPT=number_CPT,
                            split_cpt=split_cpt,
                            feature_type='gdf',
                            regressor=regressor,
                            ml_params=ml_params,
                            field=field,
                            x_values=x_values,
                            y_values=y_values
                            )


field_predicted_1 = Field2D(y_predicted_1.reshape(len(y_values), len(x_values)), "XY GradientBoost")
field_predicted_2 = Field2D(y_predicted_2.reshape(len(y_values), len(x_values)), "GDF GradientBoost")

plot_field_comparison([field, field_predicted_1, field_predicted_2], x_values, y_values)


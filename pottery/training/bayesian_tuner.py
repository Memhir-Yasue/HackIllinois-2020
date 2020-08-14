# from ax import optimize
from custom_managed_loop import optimize


def param_tuning(iteration, evaluation_function):
    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {
                "name": "n_estimators",
                "type": "range",
                "bounds": [100, 500],
            },
            {
                "name": "eta",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "gamma",
                "type": "range",
                "bounds": [1, 3],
            },
            {
                "name": "max_depth",
                "type": "range",
                "bounds": [6, 8],
            },
            {
                "name": "min_child_weight",
                "type": "range",
                "bounds": [1, 3],
            },
            {
                "name": "max_delta_step",
                "type": "range",
                "bounds": [0.0, 10.0],
            },

            {
                "name": "subsample",
                "type": "range",
                "bounds": [0.8, 1.0],
            },
            {
                "name": "colsample_bytree",
                "type": "range",
                "bounds": [0.01, 1.0],
            },
            {
                "name": "colsample_bylevel",
                "type": "range",
                "bounds": [0.01, 1.0],
            },
            {
                "name": "lambda",
                "type": "range",
                "bounds": [0., 4000.],
            },
            {
                "name": "alpha",
                "type": "range",
                "bounds": [0, 10],
            },
            {
                "name": "random_state",
                "type": "fixed",
                "value": 42,
            },
            ],
            # Booth function
            evaluation_function=evaluation_function,
            minimize=True,
            random_seed=42,
            total_trials=iteration,
        )
    return best_parameters, best_values





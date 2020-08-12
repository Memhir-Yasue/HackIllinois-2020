from ax import optimize


def param_tuning(iteration, evaluation_function):
    best_parameters, best_values, experiment, model = optimize(
        parameters=[
              {
                "name": "n_estimators",
                "type": "range",
                "bounds": [100, 1000],
              },
              {
                "name": "max_depth",
                "type": "range",
                "bounds": [6, 50],
              },
              {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.01, 1.0],
              },
            ],
            # Booth function
            evaluation_function=evaluation_function,
            minimize=True,
            random_seed=42,
            total_trials=iteration,
        )
    return best_parameters, best_values



import logging
from typing import List, Optional, Tuple

from ax.core import Arm
from ax.utils.common.logger import get_logger
from ax.service.managed_loop import OptimizationLoop

from ax.core.experiment import Experiment, SearchSpace
from ax.core.simple_experiment import SimpleExperiment, TEvaluationFunction
from ax.core.types import TModelPredictArm, TParameterization
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features

from ax.service.utils.instantiation import (
    TParameterRepresentation,
    constraint_from_str,
    outcome_constraint_from_str,
    parameter_from_json,
)


logger: logging.Logger = get_logger(__name__)


class customOptimizationLoop(OptimizationLoop):

    def __init__(self, experiment, total_trials, arms_per_trial, random_seed, wait_time, generation_strategy):

        super().__init__(experiment=experiment,
                         total_trials=total_trials,
                         arms_per_trial=arms_per_trial,
                         random_seed=random_seed,
                         wait_time=wait_time,
                         generation_strategy=generation_strategy,)

    def run_trial(self) -> None:
        """Run a single step of the optimization plan."""
        if self.current_trial >= self.total_trials:
            raise ValueError("Optimization is complete, cannot run another trial.")
        if self.current_trial == 0:
            logger.info(f" ****ADDING INITIAL PARAMS **** {self.current_trial}...")
            self.experiment.new_trial().add_arm(Arm(name='initial', parameters={'n_estimators': 100,
                                                                                'max_depth': 6,
                                                                                'eta': 0.3,
                                                                                'gamma': 0.0,
                                                                                'max_delta_step':0.0,
                                                                                'min_child_weight':5.0,
                                                                                'subsample': 1.0,
                                                                                'colsample_bytree':1.0,
                                                                                'colsample_bylevel':1.0,
                                                                                'lambda':1.0,
                                                                                'alpha':0.0,
                                                                                'random_state': 42}))
            # self.experiment.trials[0].run()
        logger.info(f"Running optimization trial {self.current_trial + 1}...")
        arms_per_trial = self.arms_per_trial
        if arms_per_trial == 1:
            trial = self.experiment.new_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment,
                    pending_observations=get_pending_observation_features(
                        experiment=self.experiment
                    ),
                )
            )
        elif arms_per_trial > 1:
            trial = self.experiment.new_batch_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment, n=arms_per_trial
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"Invalid number of arms per trial: {arms_per_trial}")
        trial.fetch_data()
        self.current_trial += 1

    def full_run(self) -> "customOptimizationLoop":
        """Runs full optimization loop as defined in the provided optimization
        plan."""
        num_steps = self.total_trials
        logger.info(f"Started full custom optimization with {num_steps} steps.")
        for _ in range(num_steps):
            self.run_trial()
        return self

    @staticmethod
    def with_custom_evaluation_function(
        parameters: List[TParameterRepresentation],
        evaluation_function: TEvaluationFunction,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
        random_seed: Optional[int] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
    ) -> "customOptimizationLoop":
        """Constructs a synchronous `OptimizationLoop` using an evaluation
        function."""
        exp_parameters = [parameter_from_json(p) for p in parameters]
        parameter_map = {p.name: p for p in exp_parameters}
        experiment = SimpleExperiment(
            name=experiment_name,
            search_space=SearchSpace(
                parameters=exp_parameters,
                parameter_constraints=None
                if parameter_constraints is None
                else [
                    constraint_from_str(c, parameter_map) for c in parameter_constraints
                ],
            ),
            objective_name=objective_name,
            evaluation_function=evaluation_function,
            minimize=minimize,
            outcome_constraints=[
                outcome_constraint_from_str(c) for c in (outcome_constraints or [])
            ],
        )
        return customOptimizationLoop(
            experiment=experiment,
            total_trials=total_trials,
            arms_per_trial=arms_per_trial,
            random_seed=random_seed,
            wait_time=wait_time,
            generation_strategy=generation_strategy,
        )


def optimize(
        parameters: List[TParameterRepresentation],
        evaluation_function: TEvaluationFunction,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        random_seed: Optional[int] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
) -> Tuple[
    TParameterization, Optional[TModelPredictArm], Experiment, Optional[ModelBridge]
]:
    """Construct and run a full optimization loop."""
    loop = customOptimizationLoop.with_custom_evaluation_function(
        parameters=parameters,
        objective_name=objective_name,
        evaluation_function=evaluation_function,
        experiment_name=experiment_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints,
        total_trials=total_trials,
        arms_per_trial=arms_per_trial,
        random_seed=random_seed,
        generation_strategy=generation_strategy,
    )
    loop.full_run()
    parameterization, values = loop.get_best_point()
    return parameterization, values, loop.experiment, loop.get_current_model()

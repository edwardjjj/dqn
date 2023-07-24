import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import A2C, PPO
import optuna
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
from stable_baselines3.common.callbacks import EvalCallback

# env_id = "Pendulum-v1"
# eval_env = make_vec_env(env_id, n_envs=10)
# training_budget = 4000
#
# dqn_model = PPO("MlpPolicy", env_id, seed=0, verbose=0).learn(training_budget)
#
# mean_reward, std_reward = evaluate_policy(
#     dqn_model, eval_env, n_eval_episodes=100, deterministic=True
# )
#
# print(f"PPO mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")

N_TRAILS = 100
N_JOBS = 1
N_STARTUP_TRAILS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 100
TIMEOUT = int(60 * 15)
ENV_ID = "CartPole-v1"
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
   
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_a2c_params(trial))
    eval_env = make_vec_env(ENV_ID, N_EVAL_ENVS)
    model = A2C(**kwargs)

    eval_callback = TrialEvalCallback(
        eval_env, trial, N_EVAL_EPISODES, EVAL_FREQ, deterministic=True
    )
    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_env.close

    if nan_encountered:
        return float("nan")
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def main():
    torch.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRAILS)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRAILS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRAILS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass
    print("Number of finished trials: ", len(study.trials))

    print("Best, trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")

    print("  Prams:  ")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    print("  User attrs:  ")
    for key, value in trial.user_attrs.items():
        print(f"  {key}: {value}")

    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()

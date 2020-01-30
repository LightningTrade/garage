#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
import random
import tensorflow as tf
from garage.envs import normalize, normalized_reward_env
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT
from garage import wrap_experiment

MT50_envs_by_id = {
    task: env(*HARD_MODE_ARGS_KWARGS[type][task]['args'],
              **HARD_MODE_ARGS_KWARGS[type][task]['kwargs'])
    for type in ['train', 'test']
    for (task, env) in HARD_MODE_CLS_DICT[type].items()
}

env_ids = ['reach-v1',
           'push-v1',
           'pick-place-v1',
           'reach-wall-v1',
           'pick-place-wall-v1',
           'push-wall-v1',
           'door-open-v1',
           'door-close-v1',
           'drawer-open-v1',
           'drawer-close-v1',
           'button-press_topdown-v1',
           'button-press-v1',
           'button-press-topdown-wall-v1',
           'button-press-wall-v1',
           'peg-insert-side-v1',
           'peg-unplug-side-v1',
           'window-open-v1',
           'window-close-v1',
           'dissassemble-v1',
           'hammer-v1',
           'plate-slide-v1',
           'plate-slide-side-v1',
           'plate-slide-back-v1',
           'plate-slide-back-side-v1',
           'handle-press-v1',
           'handle-pull-v1',
           'handle-press-side-v1',
           'handle-pull-side-v1',
           'stick-push-v1',
           'stick-pull-v1',
           'basket-ball-v1',
           'soccer-v1',
           'faucet-open-v1',
           'faucet-close-v1',
           'coffee-push-v1',
           'coffee-pull-v1',
           'coffee-button-v1',
           'sweep-v1',
           'sweep-into-v1',
           'pick-out-of-hole-v1',
           'assembly-v1',
           'shelf-place-v1',
           'push-back-v1',
           'lever-pull-v1',
           'dial-turn-v1',
           'bin-picking-v1',
           'box-close-v1',
           'hand-insert-v1',
           'door-lock-v1',
           'door-unlock-v1']
# env_ids = ['push-v1']
# env_ids = ['reach-v1']
# env_ids = ['pick-place-v1']

MT50_envs = [TfEnv(normalized_reward_env(MT50_envs_by_id[i], normalize_reward=True)) for i in env_ids]


@wrap_experiment
def trpo_mt50(ctxt=None, seed=1):

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MultiEnvWrapper(MT50_envs, env_ids, sample_strategy=round_robin_strategy)

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
            ),
        )

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=150,
                    discount=0.99,
                    gae_lambda=0.97,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=1500, batch_size=len(MT50_envs)*10*150)


seeds = random.sample(range(100), 1)
for seed in seeds:
    trpo_mt50(seed=seed)

"""
Script to operate DQN agents trained with Prioritized Experience Replay.
"""

import argparse

import torch as tc
import gym

from per.agents.architectures import NatureCNN
from per.agents.heads import LinearActionValueHead, DuelingActionValueHead
from per.agents.dqn import QNetwork
from per.algos.replay import PrioritizedReplayMemory
from per.algos.q_learning import training_loop


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of Prioritized Experience Replay")

    ### mode, environment, algo, architecture
    parser.add_argument("--mode", choices=["train", "evaluate", "video"])
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4')
    parser.add_argument("--double_dqn", choices=[0,1], default=1)
    parser.add_argument("--dueling_head", choices=[0,1], default=0)

    ### training hparams
    parser.add_argument("--optimizer_name", choices=['rmsprop', 'adam', 'sgd'], default='rmsprop')
    parser.add_argument("--learning_rate", type=float, default=0.00008)
    parser.add_argument("--huber_loss", choices=[0,1], default=1)
    parser.add_argument("--target_update_interval", type=int, default=1e4)
    parser.add_argument("--max_env_steps_per_process", type=int, default=50e6)
    parser.add_argument("--num_env_steps_per_policy_update", type=int, default=4)
    parser.add_argument("--num_env_steps_before_learning", type=int, default=5e4)
    parser.add_argument("--batches_per_policy_update", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_init", type=float, default=1.0)
    parser.add_argument("--epsilon_final", type=float, default=0.1)
    parser.add_argument("--epsilon_annealing", choice=[0,1], default=1)
    parser.add_argument("--epsilon_annealing_start_step", type=int, default=0)
    parser.add_argument("--epsilon_annealing_end_step", type=int, default=1e6)

    ### replay hparams
    parser.add_argument("--replay_memory_size", type=int, default=1e6)
    parser.add_argument("--alpha_init", type=float, default=0.6)
    parser.add_argument("--beta_init", type=float, default=0.4)
    parser.add_argument("--alpha_annealing", choice=[0,1], default=0)
    parser.add_argument("--beta_annealing", choice=[0,1], default=1)
    parser.add_argument("--alpha_annealing_start_step", type=int, default=0)
    parser.add_argument("--beta_annealing_start_step", type=int, default=0)
    return parser


def create_net(num_actions, dueling_head):
    architecture = NatureCNN()
    if dueling_head:
        head = DuelingActionValueHead(
            num_features=architecture.output_dim,
            num_actions=num_actions)
    else:
        head = LinearActionValueHead(
            num_features=architecture.output_dim,
            num_actions=num_actions)
    return QNetwork(
        architecture=architecture,
        head=head)


def create_optimizer(network, optimizer_name, learning_rate):
    if optimizer_name == 'rmsprop':
        return tc.optim.RMSprop(
            params=network.params(),
            lr=learning_rate,
            momentum=0.95,
            alpha=0.95,
            eps=0.01)
    if optimizer_name == 'adam':
        return tc.optim.Adam(
            params=network.params(),
            lr=learning_rate,
            eps=1e-6)
    if optimizer_name == 'sgd':
        return tc.optim.SGD(
            params=network.params(),
            lr=learning_rate)
    raise ValueError(f"Optimizer name {optimizer_name}, not supported.")


def main():
    args = create_argparser().parse_args()

    # get comm

    # create env

    # create learning system.
    q_network = create_net(
        num_actions=env.num_actions,
        dueling_head=args.dueling_head)

    target_network = create_net(
        num_actions=env.num_actions,
        dueling_head=args.dueling_head)

    optimizer = create_optimizer(
        network=q_network,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate)

    scheduler = None

    replay_memory = PrioritizedReplayMemory(capacity=



    def training_loop(
            t0: int,
            env: gym.Env,
            q_network: QNetwork,
            replay_memory: PrioritizedReplayMemory,
            optimizer: tc.optim.Optimizer,
            scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],
            target_network: QNetwork,
            target_update_interval: int,
            max_env_steps_per_process: int,
            num_env_steps_per_policy_update: int,
            num_env_steps_before_learning: int,
            batches_per_policy_update: int,
            batch_size: int,
            alpha_annealing_fn: Callable[[int, int], float],
            beta_annealing_fn: Callable[[int, int], float],
            epsilon_anneal_fn: Callable[[int, int], float],
            gamma: float,
            double_dqn: bool,
            huber_loss: bool
    ):

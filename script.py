"""
Script to operate DQN agents trained with Prioritized Experience Replay.
"""

import argparse
import os
import pickle

import torch as tc

from per.agents.architectures import NatureCNN
from per.agents.heads import LinearActionValueHead, DuelingActionValueHead
from per.agents.dqn import QNetwork
from per.algos.replay import PrioritizedReplayMemory
from per.algos.q_learning import training_loop

from per.utils.checkpoint_util import maybe_load_checkpoint, maybe_load_replay_memory
from per.utils.comm_util import get_comm, sync_state, ROOT_RANK
from per.utils.atari_util import make_atari, wrap_deepmind
from per.utils.video_util import save_video


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of Prioritized Experience Replay")

    ### mode, environment, algo, architecture
    parser.add_argument("--mode", choices=["train", "evaluate", "video"], default='train')
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4')
    parser.add_argument("--double_dqn", choices=[0,1], default=1)
    parser.add_argument("--dueling_head", choices=[0,1], default=0)

    ### training hparams
    parser.add_argument("--optimizer_name", choices=['rmsprop', 'adam', 'sgd'], default='rmsprop')
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
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
    parser.add_argument("--epsilon_annealing", choices=[0,1], default=1)
    parser.add_argument("--epsilon_annealing_start_step", type=int, default=0)
    parser.add_argument("--epsilon_annealing_end_step", type=int, default=1e6)

    ### replay hparams
    parser.add_argument("--replay_memory_size", type=int, default=1e6)
    parser.add_argument("--alpha_init", type=float, default=0.6)
    parser.add_argument("--beta_init", type=float, default=0.4)
    parser.add_argument("--alpha_annealing", choices=[0,1], default=0)
    parser.add_argument("--beta_annealing", choices=[0,1], default=1)
    parser.add_argument("--alpha_annealing_start_step", type=int, default=0)
    parser.add_argument("--beta_annealing_start_step", type=int, default=0)

    ### checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default='models_dir/')
    parser.add_argument("--run_name", type=str, default='default_hparams')
    parser.add_argument("--checkpoint_interval", type=int, default=1e4)
    parser.add_argument("--replay_checkpointing", choices=[0,1], default=1)
    parser.add_argument("--auto_load_config", choices=[0,1], default=1)
    return parser


def maybe_load_config(args, comm):
    # picks up a project from where it left off,
    # without requiring all command line args to be specified again
    if not bool(args.auto_load_config):
        return args
    mode = args.mode
    base_path = os.path.join(args.checkpoint_dir, args.run_name)
    config_path = os.path.join(base_path, 'config')
    if comm.Get_rank() == ROOT_RANK:
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                args = pickle.load(f)
            args.mode = mode
        else:
            os.makedirs(base_path, exist_ok=True)
            with open(config_path, 'wb') as f:
                pickle.dump(args, f)
    args = comm.broadcast(args, root=ROOT_RANK)
    return args


def create_env(env_name, mode):
    env = make_atari(env_name)
    env.seed(None)
    env = wrap_deepmind(
        env=env,
        frame_stack=True,
        clip_rewards=(mode == 'train'),
        episode_life=(mode == 'train'))
    env.seed(None)
    return env


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
            params=network.parameters(),
            lr=learning_rate,
            momentum=0.95,
            alpha=0.95,
            eps=0.01)
    if optimizer_name == 'adam':
        return tc.optim.Adam(
            params=network.parameters(),
            lr=learning_rate,
            eps=1e-6)
    if optimizer_name == 'sgd':
        return tc.optim.SGD(
            params=network.parameters(),
            lr=learning_rate)
    raise ValueError(f"Optimizer name {optimizer_name}, not supported.")


def create_scheduler():
    return None


def create_replay_memory(replay_memory_size, alpha_init, beta_init):
    return PrioritizedReplayMemory(
        capacity=replay_memory_size,
        alpha=alpha_init,
        beta=beta_init,
        eps=0.001)


def create_annealing_fn(
        initial_value,
        final_value,
        do_annealing,
        start_step,
        end_step
):
    def annealing_fn(t):
        if not do_annealing:
            return initial_value
        numer = (t - start_step)
        denom = (end_step - start_step)
        frac_done = max(0.0, min(numer / denom, 1.0))
        value_t = initial_value + (final_value-initial_value) * frac_done
        return value_t
    return annealing_fn


def main():
    args = create_argparser().parse_args()
    args = maybe_load_config(args)
    comm = get_comm()

    ### create env.
    env = create_env(
        env_name=args.env_name,
        mode=args.mode)

    ### create learning system.
    q_network = create_net(
        num_actions=env.action_space.n,
        dueling_head=args.dueling_head)

    target_network = create_net(
        num_actions=env.action_space.n,
        dueling_head=args.dueling_head)

    optimizer = create_optimizer(
        network=q_network,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate)

    scheduler = create_scheduler()

    replay_memory = create_replay_memory(
        replay_memory_size=args.replay_memory_size,
        alpha_init=args.alpha_init,
        beta_init=args.beta_init)

    ### load checkpoint, if applicable.
    num_env_steps_thus_far = 0
    if comm.Get_rank() == ROOT_RANK:
        num_env_steps_thus_far = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            scheduler=scheduler,
            steps=None)

    if args.mode == 'train':
        ### sync state.
        num_env_steps_thus_far = comm.bcast(num_env_steps_thus_far, root=ROOT_RANK)
        sync_state(
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            scheduler=scheduler,
            comm=comm,
            root=ROOT_RANK)

        ### maybe load replay memory on each process.
        if bool(args.replay_checkpointing):
            replay_memory = maybe_load_replay_memory(
                steps=num_env_steps_thus_far,
                checkpoint_dir=args.checkpoint_dir,
                run_name=args.run_name,
                rank=comm.Get_rank(),
                replay_memory=replay_memory)

        ### create annealing functions.
        alpha_annealing_fn = create_annealing_fn(
            initial_value=args.alpha_init,
            final_value=0.0,
            do_annealing=bool(args.alpha_annealing),
            start_step=args.alpha_annealing_start_step,
            end_step=args.max_env_steps_per_process)
        beta_annealing_fn = create_annealing_fn(
            initial_value=args.beta_init,
            final_value=1.0,
            do_annealing=bool(args.beta_annealing),
            start_step=args.beta_annealing_start_step,
            end_step=args.max_env_steps_per_process)
        epsilon_annealing_fn = create_annealing_fn(
            initial_value=args.epsilon_init,
            final_value=args.epsilon_final,
            do_annealing=bool(args.epsilon_annealing),
            start_step=args.epsilon_annealing_start_step,
            end_step=args.epsilon_annealing_end_step)

        ### run it!
        training_loop(
            env=env,
            q_network=q_network,
            replay_memory=replay_memory,
            optimizer=optimizer,
            scheduler=scheduler,
            target_network=target_network,
            target_update_interval=int(args.target_update_interval),
            max_env_steps_per_process=int(args.max_env_steps_per_process),
            num_env_steps_per_policy_update=int(args.num_env_steps_per_policy_update),
            num_env_steps_before_learning=int(args.num_env_steps_before_learning),
            num_env_steps_thus_far=num_env_steps_thus_far,
            batches_per_policy_update=args.batches_per_policy_update,
            batch_size=args.batch_size,
            alpha_annealing_fn=alpha_annealing_fn,
            beta_annealing_fn=beta_annealing_fn,
            epsilon_annealing_fn=epsilon_annealing_fn,
            gamma=args.discount_gamma,
            double_dqn=bool(args.double_dqn),
            huber_loss=bool(args.huber_loss),
            comm=comm,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            checkpoint_interval=args.checkpoint_interval,
            replay_checkpointing=bool(args.replay_checkpointing))

    elif args.mode == 'video':
        if comm.Get_rank() == ROOT_RANK:
            save_video(
                checkpoint_dir=args.checkpoint_dir,
                run_name=args.run_name,
                q_network=q_network,
                env=env)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

from typing import Tuple, List, Optional, Callable

import torch as tc
import numpy as np
import gym
from mpi4py import MPI

from per.agents.dqn import QNetwork
from per.algos.replay import ExperienceTuple, PrioritizedReplayMemory
from per.utils.comm_util import sync_grads


def mod_check(
        step: int,
        min_step: int,
        mod: int
) -> bool:
    return step >= min_step and step % mod == 0


@tc.no_grad()
def update_target_network(
        q_network: QNetwork,
        target_network: QNetwork
) -> None:
    for dest, src in zip(target_network.parameters(), q_network.parameters()):
        dest.copy_(src)


@tc.no_grad()
def step_env(
        q_network: QNetwork,
        epsilon: float,
        o_t: np.ndarray,
        env: gym.Env
) -> ExperienceTuple:
    a_t = q_network.sample(
        x=tc.FloatTensor(o_t).unsqueeze(0),
        epsilon=epsilon)
    a_t = a_t.squeeze(0).detach().numpy()

    o_tp1, r_t, d_t, _ = env.step(action=a_t)
    if d_t:
        o_tp1 = env.reset()
    d_t = float(d_t)

    experience_tuple_t = ExperienceTuple(
        s_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t, s_tp1=o_tp1, td_err=None)
    return experience_tuple_t


def get_tensor(
        experience_tuples: List[ExperienceTuple],
        field_name: str,
        dtype: str
) -> tc.Tensor:
    lt = list(map(lambda et: getattr(et, field_name), experience_tuples))
    if dtype == 'float':
        return tc.FloatTensor(lt)
    if dtype == 'long':
        return tc.LongTensor(lt)
    raise ValueError('Unsupported dtype for function extract_field.')


def compute_qvalues_and_targets(
        experience_tuples: List[ExperienceTuple],
        q_network: QNetwork,
        target_network: QNetwork,
        gamma: float,
        double_dqn: bool
) -> Tuple[tc.Tensor, tc.Tensor]:
    o_t = get_tensor(experience_tuples, 's_t', 'float')
    a_t = get_tensor(experience_tuples, 'a_t', 'long')
    r_t = get_tensor(experience_tuples, 'r_t', 'float')
    d_t = get_tensor(experience_tuples, 'd_t', 'float')
    o_tp1 = get_tensor(experience_tuples, 's_tp1', 'float')
    if double_dqn:
        qs_tp1_tgt = target_network(o_tp1)
        qs_tp1 = q_network(o_tp1)
        argmax_a_tp1 = tc.argmax(qs_tp1, dim=-1)
        q_tp1_tgt = tc.gather(input=qs_tp1_tgt, dim=-1, index=argmax_a_tp1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t)
    else:
        qs_tp1_tgt = target_network(o_tp1)
        q_tp1_tgt = tc.max(qs_tp1_tgt, dim=-1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t)
    return q_t, y_t


def compute_losses(
        inputs: tc.Tensor,
        targets: tc.Tensor,
        weights: tc.Tensor,
        huber_loss: bool
) -> tc.FloatTensor:
    if huber_loss:
        mb_loss_terms = tc.nn.SmoothL1Loss()(
            inputs=inputs,
            targets=targets,
            reduction='none')
    else:
        mb_loss_terms = tc.nn.MSELoss()(
            inputs=inputs,
            targets=targets,
            reduction='none')
    mb_loss = tc.sum(weights * mb_loss_terms)
    return mb_loss.float()


def training_loop(
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
        num_env_steps_thus_far: int,
        batches_per_policy_update: int,
        batch_size: int,
        alpha_annealing_fn: Callable[[int], float],
        beta_annealing_fn: Callable[[int], float],
        epsilon_annealing_fn: Callable[[int], float],
        gamma: float,
        double_dqn: bool,
        huber_loss: bool,
        comm: type(MPI.COMM_WORLD),
        checkpoint_fn: Callable[[int], None],
        checkpoint_interval: int
) -> None:
    o_t = env.reset()
    for t in range(num_env_steps_thus_far, max_env_steps_per_process):
        ### maybe update target network.
        if mod_check(t, num_env_steps_before_learning, target_update_interval):
             update_target_network(
                q_network=q_network,
                target_network=target_network)

        ### act.
        experience_tuple_t = step_env(
            q_network=q_network,
            epsilon=epsilon_annealing_fn(t),
            o_t=o_t,
            env=env)
        o_t = experience_tuple_t.s_tp1

        ### update replay memory.
        replay_memory.update_alpha(alpha_annealing_fn(t))
        replay_memory.update_beta(beta_annealing_fn(t))
        replay_memory.insert(experience_tuple_t)

        ### maybe learn.
        if mod_check(t, num_env_steps_before_learning, num_env_steps_per_policy_update):
            for _ in range(batches_per_policy_update):
                samples = replay_memory.sample(batch_size=batch_size)
                qs, ys = compute_qvalues_and_targets(
                    experience_tuples=samples['data'],
                    q_network=q_network,
                    target_network=target_network,
                    gamma=gamma,
                    double_dqn=double_dqn)

                deltas = ys - qs
                replay_memory.update_td_errs(
                    indices=samples['indices'],
                    td_errs=list(deltas.detach().numpy()))

                ws = tc.FloatTensor(samples['weights'])
                mb_loss = compute_losses(
                    inputs=qs,
                    targets=ys,
                    weights=ws,
                    huber_loss=huber_loss)
                optimizer.zero_grad()
                mb_loss.backward()
                sync_grads(model=q_network, comm=comm)
                optimizer.step()
                if scheduler:
                    scheduler.step()

        if mod_check(t, num_env_steps_before_learning, checkpoint_interval):
            checkpoint_fn(t+1)

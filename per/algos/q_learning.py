from typing import Tuple, List, Optional, Callable

import torch as tc
import numpy as np
import gym
from mpi4py import MPI

from per.agents.dqn import QNetwork
from per.algos.replay import ExperienceTuple, PrioritizedReplayMemory
from per.utils.comm_util import sync_grads, ROOT_RANK
from per.utils.checkpoint_util import save_checkpoint, save_replay_memory


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
    a_t = int(a_t.squeeze(0).detach().numpy())

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
        q_tp1_tgt = tc.gather(
            input=qs_tp1_tgt, dim=-1, index=argmax_a_tp1.unsqueeze(-1)).squeeze(-1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t.unsqueeze(-1)).squeeze(-1)
    else:
        qs_tp1_tgt = target_network(o_tp1)
        q_tp1_tgt = tc.max(qs_tp1_tgt, dim=-1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t.unsqueeze(-1)).squeeze(-1)
    return q_t, y_t


def compute_loss(
        inputs: tc.Tensor,
        targets: tc.Tensor,
        weights: tc.Tensor,
        huber_loss: bool
) -> tc.Tensor:
    if huber_loss:
        criterion = tc.nn.SmoothL1Loss(reduction='none')
    else:
        criterion = tc.nn.MSELoss(reduction='none')
    loss_terms = criterion(input=inputs, target=targets)
    loss = tc.sum(weights * loss_terms)
    return loss


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
        checkpoint_dir: str,
        run_name: str,
        checkpoint_interval: int,
        replay_checkpointing: bool,
        verbose=True
) -> None:

    if num_env_steps_thus_far == 0:
        update_target_network(
            q_network=q_network,
            target_network=target_network)
    o_t = env.reset()

    for t in range(num_env_steps_thus_far, max_env_steps_per_process):
        global_t = t * comm.Get_size()
        if verbose and comm.Get_rank() == ROOT_RANK:
            print(global_t)
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
                loss = compute_loss(
                    inputs=qs,
                    targets=ys,
                    weights=ws,
                    huber_loss=huber_loss)
                optimizer.zero_grad()
                loss.backward()
                sync_grads(model=q_network, comm=comm)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                loss_np = loss.detach().numpy()
                loss_sum = comm.allreduce(loss_np, op=MPI.SUM)
                loss_mean = loss_sum / comm.Get_size()
                if comm.Get_rank() == ROOT_RANK:
                    print(f"global timestep: {global_t}... loss: {loss_mean}")

        ### maybe save checkpoint.
        if mod_check(t, num_env_steps_before_learning, checkpoint_interval):
            if comm.Get_rank() == ROOT_RANK:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    run_name=run_name,
                    steps=t+1,
                    q_network=q_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    scheduler=scheduler)
            if replay_checkpointing:
                save_replay_memory(
                    checkpoint_dir=checkpoint_dir,
                    run_name=run_name,
                    rank=comm.Get_rank(),
                    steps=t+1,
                    replay_memory=replay_memory)

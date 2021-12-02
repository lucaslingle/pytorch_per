from typing import List, Optional, Callable

import torch as tc
import gym

from per.agents.dqn import QNetwork
from per.algos.replay import ExperienceTuple, PrioritizedReplayMemory


@tc.no_grad()
def update_target_network(
        q_network: QNetwork,
        target_network: QNetwork
):
    for dest, src in zip(target_network.parameters(), q_network.parameters()):
        dest.copy_(src)


def compute_td_errs(
        q_network: QNetwork,
        target_network: QNetwork,
        o_t: tc.FloatTensor,
        a_t: tc.LongTensor,
        r_t: tc.FloatTensor,
        d_t: tc.FloatTensor,
        o_tp1: tc.FloatTensor,
        gamma: float,
        double_dqn: bool
):
    if double_dqn:
        qs_tp1_tgt = target_network(o_tp1)
        qs_tp1 = q_network(o_tp1)
        argmax_a_tp1 = tc.argmax(qs_tp1, dim=-1)
        q_tp1_tgt = tc.gather(input=qs_tp1_tgt, dim=-1, index=argmax_a_tp1)
        y_t = r_t + (1. - d_t) * gamma * q_tp1_tgt

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t)

        td_errs = y_t.detach() - q_t
    else:
        qs_tp1_tgt = target_network(o_tp1)
        q_tp1_tgt = tc.max(qs_tp1_tgt, dim=-1)
        y_t = r_t + (1. - d_t) * gamma * q_tp1_tgt

        qs_t = q_network(o_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t)

        td_errs = y_t.detach() - q_t
    return td_errs


def extract_field(
        experience_tuples: List[ExperienceTuple],
        field_name: str,
        dtype: str
):
    assert dtype in ['float', 'long']
    lt = list(map(lambda et: getattr(et, field_name), experience_tuples))
    tn = tc.stack(lt, dim=0)
    if dtype == 'float':
        return tc.FloatTensor(tn)
    if dtype == 'long':
        return tc.LongTensor(tn)
    raise ValueError('Unsupported dtype for function extract_field.')


def generate_experience(
        t0: int,
        env: gym.Env,
        q_network: QNetwork,
        target_network: QNetwork,
        max_env_steps_per_process: int,
        epsilon_anneal_fn: Callable[[int, int], float],
        gamma: float,
        double_dqn: bool
):
    t = t0
    o_t = env.reset()
    while t < max_env_steps_per_process:
        ### act.
        epsilon_t = epsilon_anneal_fn(t, max_env_steps_per_process)
        a_t = q_network.sample(
            x=tc.FloatTensor(o_t).unsqueeze(0),
            epsilon=epsilon_t)
        a_t = a_t.squeeze(0).detach().numpy()

        o_tp1, r_t, d_t, _ = env.step(action=a_t)
        if d_t:
            o_tp1 = env.reset()
        d_t = float(d_t)

        ### compute td error.
        td_err = compute_td_errs(
            q_network=q_network,
            target_network=target_network,
            o_t=tc.FloatTensor(o_t).unsqueeze(0),
            a_t=tc.LongTensor(a_t).unsqueeze(0),
            r_t=tc.FloatTensor(r_t).unsqueeze(0),
            d_t=tc.FloatTensor(d_t).unsqueeze(0),
            o_tp1=tc.FloatTensor(o_tp1).unsqueeze(0),
            gamma=gamma,
            double_dqn=double_dqn)
        td_err = td_err.squeeze(0).detach().numpy()

        ### create experience tuple
        experience_tuple = ExperienceTuple(
            s_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t, s_tp1=o_tp1,
            td_err=td_err)

        ### increment t and yield experience tuple
        t += 1
        yield experience_tuple


def learn(
        q_network: QNetwork,
        replay_memory: PrioritizedReplayMemory,
        optimizer: tc.optim.Optimizer,
        scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],
        target_network: QNetwork,
        batches_per_policy_update: int,
        batch_size: int,
        gamma: float,
        double_dqn: bool
):
    for _ in range(batches_per_policy_update):
        samples = replay_memory.sample(batch_size=batch_size)
        mb_td_errs = compute_td_errs(
            q_network=q_network,
            target_network=target_network,
            o_t=extract_field(samples['data'], 's_t', 'float'),
            a_t=extract_field(samples['data'], 'a_t', 'long'),
            r_t=extract_field(samples['data'], 'r_t', 'float'),
            d_t=extract_field(samples['data'], 'd_t', 'float'),
            o_tp1=extract_field(samples['data'], 's_tp1', 'float'),
            gamma=gamma,
            double_dqn=double_dqn)

        replay_memory.update_td_errs(
            indices=samples['indices'],
            td_errs=list(mb_td_errs.detach().numpy()))

        mb_loss_terms = tc.nn.SmoothL1Loss()(mb_td_errs, reduction='none')
        mb_loss = tc.sum(samples['weights'] * mb_loss_terms)
        optimizer.zero_grad()
        mb_loss.backward()
        # TODO(lucaslingle): sync grads here if using manual mpi dataparallel
        optimizer.step()
        if scheduler:
            scheduler.step()


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
        batches_per_policy_update: int,
        batch_size: int,
        alpha_annealing_fn: Callable[[int, int], float],
        beta_annealing_fn: Callable[[int, int], float],
        epsilon_anneal_fn: Callable[[int, int], float],
        gamma: float,
        double_dqn: bool
):
    experience_generator = generate_experience(
        t0=t0,
        env=env,
        q_network=q_network,
        target_network=target_network,
        max_env_steps_per_process=max_env_steps_per_process,
        epsilon_anneal_fn=epsilon_anneal_fn,
        gamma=gamma,
        double_dqn=double_dqn)

    for t in range(t0, max_env_steps_per_process):
        ### maybe target network update
        if t > 0 and t % target_update_interval == 0:
            update_target_network(
                q_network=q_network,
                target_network=target_network)

        ### generate experience
        alpha_t = alpha_annealing_fn(t, max_env_steps_per_process)
        beta_t = beta_annealing_fn(t, max_env_steps_per_process)
        experience_tuple_t = next(experience_generator)

        ### update replay memory
        replay_memory.update_alpha(alpha_t)
        replay_memory.update_beta(beta_t)
        replay_memory.insert(experience_tuple_t)

        ### maybe learn
        if t > 0 and t % num_env_steps_per_policy_update == 0:
            # TODO(lucaslingle):
            #    add condition to check that replay memory
            #    has at least min entries before learning,
            #    else skip learning.
            learn(
                q_network=q_network,
                replay_memory=replay_memory,
                optimizer=optimizer,
                scheduler=scheduler,
                target_network=target_network,
                batches_per_policy_update=batches_per_policy_update,
                batch_size=batch_size,
                gamma=gamma,
                double_dqn=double_dqn)

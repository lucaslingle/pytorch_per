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


def compute_td_errs(
        experience_tuples: List[ExperienceTuple],
        q_network: QNetwork,
        target_network: QNetwork,
        gamma: float,
        double_dqn: bool
):
    o_t = extract_field(experience_tuples, 's_t', 'float')
    a_t = extract_field(experience_tuples, 'a_t', 'long')
    r_t = extract_field(experience_tuples, 'r_t', 'float')
    d_t = extract_field(experience_tuples, 'd_t', 'float')
    o_tp1 = extract_field(experience_tuples, 's_tp1', 'float')
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
    o_t = env.reset()
    for t in range(t0, max_env_steps_per_process):
        ### maybe update target network
        if t > 0 and t % target_update_interval == 0:
            update_target_network(
                q_network=q_network,
                target_network=target_network)

        ### act.
        epsilon_t = epsilon_anneal_fn(t, max_env_steps_per_process)
        a_t = q_network.sample(
            x=tc.FloatTensor(o_t).unsqueeze(0), epsilon=epsilon_t)
        a_t = a_t.squeeze(0).detach().numpy()

        o_tp1, r_t, d_t, _ = env.step(action=a_t)
        if d_t:
            o_tp1 = env.reset()
        d_t = float(d_t)

        ### update replay memory.
        alpha_t = alpha_annealing_fn(t, max_env_steps_per_process)
        beta_t = beta_annealing_fn(t, max_env_steps_per_process)
        replay_memory.update_alpha(alpha_t)
        replay_memory.update_beta(beta_t)

        experience_tuple_t = ExperienceTuple(
            s_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t, s_tp1=o_tp1, td_err=None)
        replay_memory.insert(experience_tuple_t)

        ### maybe learn.
        if t > 0 and t % num_env_steps_per_policy_update == 0:
            # TODO(lucaslingle):
            #      check replay memory has at least min entries before learning,
            for _ in range(batches_per_policy_update):
                samples = replay_memory.sample(batch_size=batch_size)
                mb_td_errs = compute_td_errs(
                    experience_tuples=samples['data'],
                    q_network=q_network,
                    target_network=target_network,
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

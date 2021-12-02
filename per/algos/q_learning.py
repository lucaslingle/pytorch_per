from typing import Optional, Callable

import torch as tc
import gym

from per.agents.dqn import QNetwork
from per.algos.replay import ExperienceTuple, PrioritizedReplayMemory


def compute_td_errs(
        q_network, target_network, mb_o_t, mb_a_t, mb_r_t, gamma, mb_o_tp1,
        double_dqn
):
    if double_dqn:
        mb_qs_t = q_network(mb_o_t)
        mb_q_t = tc.gather(input=mb_qs_t, dim=-1, index=mb_a_t)

        mb_qs_tp1_tgt = target_network(mb_o_tp1)
        mb_qs_tp1 = q_network(mb_o_tp1)
        mb_argmax_a_tp1 = tc.argmax(mb_qs_tp1, dim=-1)
        mb_q_tp1_tgt = tc.gather(
            input=mb_qs_tp1_tgt, dim=-1, index=mb_argmax_a_tp1)

        mb_y_t = mb_r_t + gamma * mb_q_tp1_tgt
        mb_td_err = mb_y_t.detach() - mb_q_t
    else:
        mb_qs_t = q_network(mb_o_t)
        mb_q_t = tc.gather(input=mb_qs_t, dim=-1, index=mb_a_t)

        mb_qs_tp1_tgt = target_network(mb_o_tp1)
        mb_q_tp1_tgt = tc.max(mb_qs_tp1_tgt, dim=-1)

        mb_y_t = mb_r_t + gamma * mb_q_tp1_tgt
        mb_td_err = mb_y_t.detach() - mb_q_t
    return mb_td_err


def extract_field(experience_tuples, field_name, dtype):
    assert dtype in ['float', 'long']
    lt = list(map(lambda et: getattr(et, field_name), experience_tuples))
    tn = tc.stack(lt, dim=0)
    if dtype == 'float':
        return tc.FloatTensor(tn)
    if dtype == 'long':
        return tc.LongTensor(tn)
    raise ValueError(['Unsupported dtype for function extract_field.'])


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
        batches_per_policy_update: int,
        batch_size: int,
        alpha_annealing_fn: Callable[[int, int], float],
        beta_annealing_fn: Callable[[int, int], float],
        epsilon_anneal_fn: Callable[[int, int], float],
        gamma: float,
        double_dqn: bool
):
    t = 0
    o_t = env.reset()
    while t < max_env_steps_per_process:
        for _ in range(num_env_steps_per_policy_update):
            ### update annealed constants.
            alpha_t = alpha_annealing_fn(t, max_env_steps_per_process)
            beta_t = beta_annealing_fn(t, max_env_steps_per_process)
            eps_t = epsilon_anneal_fn(t, max_env_steps_per_process)

            ### act.
            a_t, q_t = q_network.sample(
                x=tc.FloatTensor(o_t).unsqueeze(0),
                epsilon=eps_t)
            o_tp1, r_t, done_t, _ = env.step(
                action=a_t.squeeze(0).detach().numpy())
            if done_t:
                o_tp1 = env.reset()

            ### compute td error.
            td_err = compute_td_errs(
                q_network=q_network,
                target_network=target_network,
                mb_o_t=tc.FloatTensor(o_t).unsqueeze(0),
                mb_a_t=tc.FloatTensor(a_t).unsqueeze(0),
                mb_r_t=tc.FloatTensor(r_t).unsqueeze(0),
                gamma=gamma,
                mb_o_tp1=tc.FloatTensor(o_tp1).unsqueeze(0),
                double_dqn=double_dqn)
            td_err = td_err.squeeze(0).detach().numpy()

            ### add experience tuple to replay memory.
            experience_tuple = ExperienceTuple(
                s_t=o_t, a_t=a_t, r_t=r_t, s_tp1=o_tp1, td_err=td_err)

            replay_memory.insert(experience_tuple)

        for _ in range(batches_per_policy_update):
            samples = replay_memory.sample(batch_size=batch_size)
            mb_td_err = compute_td_errs(
                q_network=q_network,
                target_network=target_network,
                mb_o_t=extract_field(samples['data'], 's_t', 'float'),
                mb_a_t=extract_field(samples['data'], 'a_t', 'long'),
                mb_r_t=extract_field(samples['data'], 'r_t', 'float'),
                gamma=gamma,
                mb_o_tp1=extract_field(samples['data'], 's_tp1', 'float'),
                double_dqn=double_dqn)

            replay_memory.update_td_errs(
                idxs=samples['indices'],
                td_errs=list(mb_td_err.detach().numpy()))

            # TODO(lucaslingle):
            #   add smooth l1 loss for mb_td_err terms.
            #   note that they're differentiable w.r.t. params of q_network.

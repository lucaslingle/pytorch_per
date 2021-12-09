from typing import Tuple, List, Optional, Callable

import torch as tc
import numpy as np
import gym

from per.agents.dqn import QNetwork
from per.algos.replay import ExperienceTuple, PrioritizedReplayMemory
from per.utils.checkpoint_util import save_checkpoint


@tc.no_grad()
def update_target_network(
        q_network: QNetwork,
        target_network: QNetwork
) -> None:
    for dest, src in zip(target_network.parameters(), q_network.parameters()):
        dest.copy_(src.detach().clone())


@tc.no_grad()
def step_env(
        q_network: QNetwork,
        epsilon: float,
        s_t: np.ndarray,
        env: gym.Env,
        device: str
) -> ExperienceTuple:
    a_t = q_network.sample(
        x=tc.tensor(s_t).float().unsqueeze(0).to(device),
        epsilon=epsilon,
        device=device)
    a_t = int(a_t.squeeze(0).detach().cpu().numpy())

    s_tp1, r_t, d_t, _ = env.step(action=a_t)
    if d_t:
        s_tp1 = env.reset()
    d_t = float(d_t)

    experience_tuple_t = ExperienceTuple(
        s_t=s_t, a_t=a_t, r_t=r_t, d_t=d_t, s_tp1=s_tp1, td_err=None)
    return experience_tuple_t


def get_tensor(
        experience_tuples: List[ExperienceTuple],
        field_name: str,
        dtype: str
) -> tc.Tensor:
    lt = list(map(lambda et: getattr(et, field_name), experience_tuples))
    ts = tc.tensor(lt)
    if dtype == 'float':
        return ts.float()
    if dtype == 'long':
        return ts.long()
    raise ValueError('Unsupported dtype for function get_tensor.')


def compute_qvalues_and_targets(
        q_network: QNetwork,
        target_network: QNetwork,
        s_t: tc.Tensor,
        a_t: tc.Tensor,
        r_t: tc.Tensor,
        d_t: tc.Tensor,
        s_tp1: tc.Tensor,
        gamma: float,
        double_dqn: bool
) -> Tuple[tc.Tensor, tc.Tensor]:
    """
    :param q_network: Q-Network.
    :param target_network: Target network.
    :param s_t: Torch FloatTensor containing minibatch of current states.
    :param a_t: Torch LongTensor containing minibatch of current actions.
    :param r_t: Torch FloatTensor containing minibatch of rewards.
    :param d_t: Torch FloatTensor containing minibatch of done signals.
    :param s_tp1: Torch FloatTensor containing minibatch of next states.
    :param gamma: Discount factor.
    :param double_dqn: Use double-Q learning?
    :return: Tuple containing
        Torch FloatTensor with q-network predictions,
        Torch FloatTensor with detached action-value targets.
    """
    if double_dqn:
        qs_tp1_tgt = target_network(s_tp1)
        qs_tp1 = q_network(s_tp1)
        argmax_a_tp1 = tc.argmax(qs_tp1, dim=-1)
        q_tp1_tgt = tc.gather(
            input=qs_tp1_tgt, dim=-1, index=argmax_a_tp1.unsqueeze(-1)).squeeze(-1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(s_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t.unsqueeze(-1)).squeeze(-1)
    else:
        qs_tp1_tgt = target_network(s_tp1)
        q_tp1_tgt = tc.max(qs_tp1_tgt, dim=-1)
        y_t = (r_t + (1. - d_t) * gamma * q_tp1_tgt).detach()

        qs_t = q_network(s_t)
        q_t = tc.gather(input=qs_t, dim=-1, index=a_t.unsqueeze(-1)).squeeze(-1)
    return q_t, y_t


def compute_loss(
        inputs: tc.Tensor,
        targets: tc.Tensor,
        weights: tc.Tensor,
        huber_loss: bool
) -> tc.Tensor:
    """
    :param inputs: Torch FloatTensor containing action-value predictions
    :param targets: Torch FloatTensor containing detached action-value targets
    :param weights: Torch FloatTensor containing rescaled importance weights.
    :param huber_loss: Boolean indicating whether to use Huber/Smooth L1 loss,
        as done by Mnih et al., 2015, Schaul et al., 2015, etc.
    :return: Torch FloatTensor containing minibatch loss.
    """
    if huber_loss:
        criterion = tc.nn.SmoothL1Loss(reduction='none')
    else:
        criterion = tc.nn.MSELoss(reduction='none')
    loss_terms = criterion(input=inputs, target=targets)
    loss = tc.sum(weights * loss_terms)
    return loss


def training_step(
        q_network: QNetwork,
        target_network: QNetwork,
        optimizer: tc.optim.Optimizer,
        scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],
        replay_memory: PrioritizedReplayMemory,
        batch_size: int,
        device: str,
        discount_gamma: float,
        double_dqn: bool,
        huber_loss: bool
):
    samples = replay_memory.sample(batch_size=batch_size)
    qs, ys = compute_qvalues_and_targets(
        q_network=q_network,
        target_network=target_network,
        s_t=get_tensor(samples['data'], 's_t', 'float').to(device),
        a_t=get_tensor(samples['data'], 'a_t', 'long').to(device),
        r_t=get_tensor(samples['data'], 'r_t', 'float').to(device),
        d_t=get_tensor(samples['data'], 'd_t', 'float').to(device),
        s_tp1=get_tensor(samples['data'], 's_tp1', 'float').to(device),
        gamma=discount_gamma,
        double_dqn=double_dqn)

    deltas = ys - qs
    replay_memory.update_td_errs(
        indices=samples['indices'],
        td_errs=list(deltas.detach().cpu().numpy()))

    ws = tc.tensor(samples['weights']).float().to(device)
    loss = compute_loss(
        inputs=qs,
        targets=ys,
        weights=ws,
        huber_loss=huber_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss


def logging_step(
        t: int,
        loss: tc.Tensor
):
    loss_np = loss.detach().cpu().numpy()
    print(f"timestep: {t}... loss: {loss_np}")


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
        device: str,
        alpha_annealing_fn: Callable[[int], float],
        beta_annealing_fn: Callable[[int], float],
        epsilon_annealing_fn: Callable[[int], float],
        discount_gamma: float,
        double_dqn: bool,
        huber_loss: bool,
        checkpoint_dir: str,
        run_name: str,
        checkpoint_interval: int
) -> None:

    if num_env_steps_thus_far == 0:
        update_target_network(
            q_network=q_network,
            target_network=target_network)
    s_t = env.reset()

    for t in range(num_env_steps_thus_far, max_env_steps_per_process):
        ### act.
        experience_tuple_t = step_env(
            q_network=q_network,
            epsilon=epsilon_annealing_fn(t),
            s_t=s_t,
            env=env,
            device=device)
        s_t = experience_tuple_t.s_tp1

        ### update replay memory.
        replay_memory.update_alpha(alpha_annealing_fn(t))
        replay_memory.update_beta(beta_annealing_fn(t))
        replay_memory.insert(experience_tuple_t)

        if replay_memory.num_items >= num_env_steps_before_learning:
            ### maybe learn.
            if t % num_env_steps_per_policy_update == 0:
                for _ in range(batches_per_policy_update):
                    loss = training_step(
                        q_network=q_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        replay_memory=replay_memory,
                        batch_size=batch_size,
                        device=device,
                        discount_gamma=discount_gamma,
                        double_dqn=double_dqn,
                        huber_loss=huber_loss)

                    logging_step(t, loss)

            ### maybe update target network.
            if t % target_update_interval == 0:
                update_target_network(
                    q_network=q_network,
                    target_network=target_network)

            ### maybe save checkpoint.
            if t % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    run_name=run_name,
                    steps=t+1,
                    q_network=q_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    scheduler=scheduler)

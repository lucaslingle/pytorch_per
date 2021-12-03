"""
Utility module for saving and loading checkpoints.
"""

import os

import torch as tc


def _format_name(kind, steps):
    filename = f"{kind}_{steps}.pth"
    return filename


def _parse_name(filename):
    kind, steps = filename.split(".")[0].split("_")
    steps = int(steps)
    return {
        "kind": kind,
        "steps": steps
    }


def _latest_n_checkpoint_steps(base_path, n=5):
    steps = set(map(lambda x: _parse_name(x)['steps'], os.listdir(base_path)))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path):
    return _latest_n_checkpoint_steps(base_path, n=1)[-1]


def save_checkpoint(
        steps,
        checkpoint_dir,
        run_name,
        q_network,
        target_network,
        optimizer,
        scheduler
    ):
    """
    Saves a checkpoint of the latest model, optimizer, scheduler state.
    Also tidies up checkpoint_dir/model_name/ by keeping only last 5 ckpts.

    Args:
        steps: num steps for the checkpoint to save.
        checkpoint_dir: checkpoint dir for checkpointing.
        run_name: run name for checkpointing.
        q_network: q-network to be saved to checkpoint.
        target_network: target network to be saved to checkpoint.
        optimizer: optimizer to be saved to checkpoint.
        scheduler: scheduler to be saved to checkpoint.

    Returns:
        None
    """
    base_path = os.path.join(checkpoint_dir, run_name)
    os.makedirs(base_path, exist_ok=True)

    qnetwork_path = os.path.join(base_path, _format_name('qnetwork', steps))
    tnetwork_path = os.path.join(base_path, _format_name('tnetwork', steps))
    optim_path = os.path.join(base_path, _format_name('optimizer', steps))
    sched_path = os.path.join(base_path, _format_name('scheduler', steps))

    # save everything
    tc.save(q_network.state_dict(), qnetwork_path)
    tc.save(target_network.state_dict(), tnetwork_path)
    tc.save(optimizer.state_dict(), optim_path)
    if scheduler is not None:
        tc.save(scheduler.state_dict(), sched_path)

    # keep only last n checkpoints
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=5)
    for file in os.listdir(base_path):
        if _parse_name(file)['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, file))


def maybe_load_checkpoint(
        checkpoint_dir,
        run_name,
        q_network,
        target_network,
        optimizer,
        scheduler,
        steps
    ):
    """
    Tries to load a checkpoint from checkpoint_dir/model_name/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    Args:
        checkpoint_dir: checkpoint dir for checkpointing.
        run_name: run name for checkpointing.
        q_network: q-network to be updated from checkpoint.
        target_network: target network to be updated from checkpoint.
        optimizer: optimizer to be updated from checkpoint.
        scheduler: scheduler to be updated from checkpoint.
        steps: num steps for the checkpoint to locate. if none, use latest.

    Returns:
        number of env steps experienced by loaded checkpoint.
    """
    base_path = os.path.join(checkpoint_dir, run_name)
    try:
        if steps is None:
            steps = _latest_step(base_path)

        qnetwork_path = os.path.join(base_path, _format_name('qnetwork', steps))
        tnetwork_path = os.path.join(base_path, _format_name('tnetwork', steps))
        optim_path = os.path.join(base_path, _format_name('optimizer', steps))
        sched_path = os.path.join(base_path, _format_name('scheduler', steps))

        q_network.load_state_dict(tc.load(qnetwork_path))
        target_network.load_state_dict(tc.load(tnetwork_path))
        optimizer.load_state_dict(tc.load(optim_path))
        if scheduler is not None:
            scheduler.load_state_dict(tc.load(sched_path))

        print(f"Loaded checkpoint from {base_path}, with step {steps}.")
        print("Continuing from checkpoint.")
    except FileNotFoundError:
        print(f"Bad checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        steps = 0

    return steps

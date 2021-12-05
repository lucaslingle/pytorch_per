"""
Utility module for saving and loading checkpoints.
"""

import os
import time

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


def _latest_n_checkpoint_steps(base_path, n=5, kind=''):
    ls = os.listdir(base_path)
    fgrep = [f for f in ls if _parse_name(f)['kind'].startswith(kind)]
    steps = set(map(lambda f: _parse_name(f)['steps'], fgrep))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path, kind=''):
    return _latest_n_checkpoint_steps(base_path, n=1, kind=kind)[-1]


def _serialize_and_save_state_dict(
        steps,
        base_path,
        kind_name,
        checkpointable
):
    """
    Saves a checkpoint of the latest model, optimizer, scheduler state.
    Also tidies up checkpoint_dir/run_name/ by keeping only last 5 ckpts.

    Args:
        steps: num steps for the checkpoint to save.
        base_path: base path for checkpointing.
        kind_name: kind name of torch module being checkpointed
            (e.g., qnetwork, optimizer, etc.).
        checkpointable: torch module/optimizer/scheduler to save checkpoint for.

    Returns:
        None
    """
    os.makedirs(base_path, exist_ok=True)

    path = os.path.join(base_path, _format_name(kind_name, steps))
    tc.save(checkpointable.state_dict(), path)

    # keep only last n checkpoints
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=5, kind=kind_name)
    for fname in os.listdir(base_path):
        parsed = _parse_name(fname)
        if parsed['kind'] == kind_name and parsed['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, fname))


def _maybe_deserialize_and_load_state_dict(
        base_path,
        kind_name,
        checkpointable,
        steps
):
    """
    Tries to load a checkpoint from checkpoint_dir/model_name/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    Args:
        base_path: base path for checkpointing.
        kind_name: kind name of torch module being checkpointed
            (e.g., qnetwork, optimizer, etc.).
        checkpointable: torch module/optimizer/scheduler to save checkpoint for.
        steps: num steps for the checkpoint to locate.

    Returns:
        number of env steps experienced by loaded checkpoint.
    """
    if steps is None:
        steps = _latest_step(base_path, kind_name)
    path = os.path.join(base_path, _format_name(kind_name, steps))
    checkpointable.load_state_dict(tc.load(path))
    return steps


def save_checkpoint(
        steps,
        checkpoint_dir,
        run_name,
        q_network,
        target_network,
        optimizer,
        scheduler
):
    kind_names = ['q_network', 'target_network', 'optimizer', 'scheduler']
    checkpointables = [q_network, target_network, optimizer, scheduler]
    checkpointables = [c for c in checkpointables if c is not None]
    base_path = os.path.join(checkpoint_dir, run_name)

    for kind_name, checkpointable in zip(kind_names, checkpointables):
        _serialize_and_save_state_dict(
            steps=steps,
            base_path=base_path,
            kind_name=kind_name,
            checkpointable=checkpointable)


def maybe_load_checkpoint(
        checkpoint_dir,
        run_name,
        q_network,
        target_network,
        optimizer,
        scheduler,
        steps=None
):
    """
    Tries to load a checkpoint from checkpoint_dir/model_name/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    Args:
        checkpoint_dir: checkpoint dir for checkpointing.
        run_name: run name for checkpointing.
        q_network: q-network.
        target_network: target network.
        optimizer: optimizer.
        scheduler: optional learning rate scheduler.
        steps: optional step number. if none, uses latest.
    """

    kind_names = ['q_network', 'target_network', 'optimizer', 'scheduler']
    checkpointables = [q_network, target_network, optimizer, scheduler]
    checkpointables = [c for c in checkpointables if c is not None]
    base_path = os.path.join(checkpoint_dir, run_name)

    try:
        steps_list = []
        for kind_name, checkpointable in zip(kind_names, checkpointables):
            _steps = _maybe_deserialize_and_load_state_dict(
                steps=steps,
                base_path=base_path,
                kind_name=kind_name,
                checkpointable=checkpointable)
            steps_list.append(_steps)
        if len(set(steps_list)) != 1:
            msg = "Iterates not aligned in latest checkpoints!\n" + \
                "Delete the offending file(s) and try again."
            raise ValueError(msg)
    except FileNotFoundError:
        print(f"Bad checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        return

    print(f"Loaded checkpoint from {base_path}, with step {steps}.")
    print("Continuing from checkpoint.")


# TODO(lucaslingle):
#  1. create functions
#      _serialize_and_save_replay_memory and
#      _deserialize_and_load_replay_memory
#  2. update save_checkpoint and maybe_load_checkpoint to take replay_memory as an arg, and
#      to use these functions. The return type on maybe_load_checkpoint should be a dict with keys
#      'latest_step', 'q_network', 'target_network', 'optimizer', 'scheduler', and 'replay_memory'.

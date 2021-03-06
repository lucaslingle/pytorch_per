"""
Utility module for saving and loading checkpoints.
"""

import os
import pickle
import zlib

import torch as tc

from per.utils.comm_util import ROOT_RANK


def _format_name(kind, steps, suffix):
    filename = f"{kind}_{steps}.{suffix}"
    return filename


def _parse_name(filename):
    kind, _, steps = filename.split(".")[0].rpartition("_")
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


def _clean(base_path, kind, n=5):
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=n, kind=kind)
    for fname in os.listdir(base_path):
        parsed = _parse_name(fname)
        if parsed['kind'] == kind and parsed['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, fname))


def _serialize_and_save_state_dict(
        base_path,
        kind_name,
        steps,
        checkpointable,
        n=5
):
    """
    Saves a checkpoint of a checkpointable.
    Also tidies up base_path by keeping only last 5 ckpts.

    Args:
        base_path: base path for checkpointing.
        kind_name: kind name of torch module being checkpointed
            (e.g., q_network, optimizer, etc.).
        steps: step number for the checkpoint to save.
        checkpointable: torch module/optimizer/scheduler to save checkpoint for.
        n: int past number of checkpoints to keep.

    Returns:
        None
    """
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, _format_name(kind_name, steps, 'pth'))
    tc.save(checkpointable.state_dict(), path)
    _clean(base_path, kind_name, n=n)


def _deserialize_and_load_state_dict(
        base_path,
        kind_name,
        steps,
        checkpointable
):
    """
    Loads a checkpoint of a checkpointable.

    Args:
        base_path: base path for checkpointing.
        kind_name: kind name of torch module being loaded
            (e.g., q_network, optimizer, etc.).
        steps: step number for the checkpoint to load. if none, uses latest.
        checkpointable: torch module/optimizer/scheduler to load checkpoint for.

    Returns:
        number of env steps experienced by loaded checkpoint.
    """
    if steps is None:
        steps = _latest_step(base_path, kind_name)
    path = os.path.join(base_path, _format_name(kind_name, steps, 'pth'))
    checkpointable.load_state_dict(tc.load(path))
    return steps


def _serialize_and_save_pickleable_state(
        base_path,
        kind_name,
        steps,
        pickleable,
        n=5
):
    """
    Saves a checkpoint of a pickleable python object.
    Also tidies up base_path by keeping only last 5 ckpts.

    Args:
        base_path: base path for checkpointing.
        kind_name: kind name of python object being pickled
            (e.g., 'replay_memory').
        steps: step number for the checkpoint to save.
        pickleable: pickleable python object to persist.
        n: int number of past checkpoints to keep.

    Returns:
        None
    """
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, _format_name(kind_name, steps, 'pkl'))
    b = pickle.dumps(pickleable)
    c = zlib.compress(b, level=6)
    with open(path, 'wb') as f:
        f.write(c)
    _clean(base_path, kind_name, n=n)


def _deserialize_and_load_pickleable_state(
        base_path,
        kind_name,
        steps
):
    """
    Loads a checkpoint of a pickleable python object.

    Args:
        base_path: base path for checkpointing.
        kind_name: kind name of python object being pickled
            (e.g., 'replay_memory').
        steps: step number for the checkpoint to load. if none, uses latest.

    Returns:
        number of env steps experienced by loaded checkpoint.
    """
    if steps is None:
        steps = _latest_step(base_path, kind_name)
    path = os.path.join(base_path, _format_name(kind_name, steps, 'pkl'))
    with open(path, 'rb') as f:
        c = f.read()
    b = zlib.decompress(c)
    pickleable = pickle.loads(b)
    return pickleable


def save_checkpoint(
        checkpoint_dir,
        run_name,
        steps,
        q_network,
        target_network,
        optimizer,
        scheduler,
        n=5
):
    """
    Saves a checkpoint to checkpoint_dir/run_name/checkpoints/.

    Args:
        checkpoint_dir: checkpoint dir for checkpointing.
        run_name: run name for checkpointing.
        steps: step number for the checkpoint to save.
        q_network: q-network.
        target_network: target network.
        optimizer: optimizer.
        scheduler: optional learning rate scheduler.
        n: int number of past checkpoints to keep

    Returns:
        None
    """
    kind_names = ['q_network', 'target_network', 'optimizer', 'scheduler']
    checkpointables = [q_network, target_network, optimizer, scheduler]
    checkpointables = [c for c in checkpointables if c is not None]
    base_path = os.path.join(checkpoint_dir, run_name, 'checkpoints')

    for kind_name, checkpointable in zip(kind_names, checkpointables):
        _serialize_and_save_state_dict(
            steps=steps,
            base_path=base_path,
            kind_name=kind_name,
            checkpointable=checkpointable,
            n=n)


def maybe_load_checkpoint(
        checkpoint_dir,
        run_name,
        steps,
        q_network,
        target_network,
        optimizer,
        scheduler,
):
    """
    Tries to load a checkpoint from checkpoint_dir/run_name/checkpoints/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    Args:
        checkpoint_dir: checkpoint dir for checkpointing.
        run_name: run name for checkpointing.
        steps: step number for checkpoint to load. if none, uses latest.
        q_network: q-network.
        target_network: target network.
        optimizer: optimizer.
        scheduler: optional learning rate scheduler.

    Returns:
        step number of the recovered checkpoints, else 0.
    """
    kind_names = ['q_network', 'target_network', 'optimizer', 'scheduler']
    checkpointables = [q_network, target_network, optimizer, scheduler]
    checkpointables = [c for c in checkpointables if c is not None]
    base_path = os.path.join(checkpoint_dir, run_name, 'checkpoints')

    step_list = []
    try:
        for kind_name, checkpointable in zip(kind_names, checkpointables):
            _steps = _deserialize_and_load_state_dict(
                steps=steps,
                base_path=base_path,
                kind_name=kind_name,
                checkpointable=checkpointable)
            step_list.append(_steps)
        if len(set(step_list)) != 1:
            msg = "Iterates not aligned in latest checkpoints!\n" + \
                  "Delete the offending file(s) and try again."
            raise ValueError(msg)
    except FileNotFoundError:
        print(f"Bad torch checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        return 0
    else:
        print(f"Loaded torch checkpoint from {base_path}, with step {_steps}.")
        print("Continuing from checkpoint.")
        return _steps


def save_replay_memory(
        checkpoint_dir,
        run_name,
        rank,
        steps,
        replay_memory,
        n=1
):
    base_path = os.path.join(checkpoint_dir, run_name, 'pickles')
    _serialize_and_save_pickleable_state(
        base_path=base_path,
        kind_name=f"replay_memory_{rank}",
        steps=steps,
        pickleable=replay_memory,
        n=n)


def maybe_load_replay_memory(
        checkpoint_dir,
        run_name,
        rank,
        steps,
        replay_memory
):
    base_path = os.path.join(checkpoint_dir, run_name, 'pickles')
    try:
        replay_memory = _deserialize_and_load_pickleable_state(
            base_path=base_path,
            kind_name=f"replay_memory_{rank}",
            steps=steps)
        print(f"Loaded replay checkpoint from {base_path}, with step {steps}.")
        print("Continuing from checkpoint.")
        return replay_memory
    except FileNotFoundError:
        if rank == ROOT_RANK:
            print(f"Bad replay checkpoint or none at {base_path} with step {steps}.")
            print("Running from scratch.")
        return replay_memory

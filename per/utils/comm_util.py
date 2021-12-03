"""
Utility module for synchronizing state across processes.
"""

from mpi4py import MPI
import torch as tc
import numpy as np


def get_comm():
    comm = MPI.COMM_WORLD
    return comm


@tc.no_grad()
def sync_state(q_network, target_network, optimizer, scheduler, comm, root):
    """
    Synchronize state of a q network, target network, optimizer, and maybe scheduler,
    using MPI.

    Args:
        q_network: q network.
        target_network: target network.
        optimizer: optimizer for the q network.
        scheduler: optional lr scheduler for the model.
        comm: mpi4py comm_world object.
        root: root mpi process rank to broadcast from.

    Returns:
        None
    """
    q_network_state_dict = comm.bcast(q_network.state_dict(), root=root)
    target_network_state_dict = comm.bcast(target_network.state_dict(), root=root)
    optimizer_state_dict = comm.bcast(optimizer.state_dict(), root=root)
    if scheduler is not None:
        scheduler_state_dict = comm.bcast(scheduler.state_dict(), root=root)

    q_network.load_state_dict(q_network_state_dict)
    target_network.load_state_dict(target_network_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state_dict)


@tc.no_grad()
def sync_grads(model, comm):
    """
    Sync gradients for a model across processes using MPI.
    The resulting synchronized gradient is stored in the p.grad field within
    each model parameter p. The stored value is the average.

    This allows us to do data-parallel training using multiple processes,
    as the p.grad fields are passed to the optimizer on each process,
    keeping the optimizer states synchronized as well.

    Args:
        model: model.
        comm: mpi4py comm_world object.

    Returns:
        None
    """
    for p in model.parameters():
        p_grad_local = p.grad.numpy()
        p_grad_global = np.zeros_like(p_grad_local)
        comm.Allreduce(sendbuf=p_grad_local, recvbuf=p_grad_global, op=MPI.SUM)
        p.grad.copy_(tc.FloatTensor(p_grad_global) / comm.Get_size())

from typing import Optional

import numpy as np


class ExperienceTuple:
    def __init__(self, s_t, a_t, r_t, d_t, s_tp1, td_err):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = r_t
        self.d_t = d_t
        self.s_tp1 = s_tp1
        self.td_err = td_err


class PrioritizedExperienceTuple:
    def __init__(
            self,
            experience_tuple: Optional[ExperienceTuple],
            summed_priority: float,
            max_priority: float
    ):
        """
        A node in the sumtree datastructure.

        :param experience_tuple: experience tuple in leaf nodes, else None.
        :param summed_priority: priority for replay in leaf nodes, else sum thereof.
        :param max_priority: priority in leaf nodes, else max thereof.
        """
        self.experience_tuple = experience_tuple
        self.summed_priority = summed_priority
        self.max_priority = max_priority


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha, beta, eps):
        self._capacity = self._get_capacity(capacity)
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._total_steps = 0

        self._tree_size = 2 * self._capacity - 1
        self._sumtree = [None for _ in range(self._tree_size)]
        self._expiration_idx = self._tree_size - self._capacity
        self._initialize_max_priority()

    def _get_capacity(self, capacity):
        # computes the number of leaf nodes to be used as replay.
        # we make the capacity a power of two to simplify the implementation.
        return 2 ** int(np.ceil(np.log(capacity) / np.log(2.0)))

    def _initialize_max_priority(self):
        # initialize the max priority to 1.0 as per Schaul et al., 2015
        self._sumtree[0] = PrioritizedExperienceTuple(
            experience_tuple=None, summed_priority=0., max_priority=1.)

    def _get_priority(self, experience_tuple):
        # computes priority using proportional prioritization.
        return (np.fabs(experience_tuple.td_err) + self._eps) ** self._alpha

    def _update_priorities(self, idx):
        # recomputes priorities for the nodes above index
        while idx != 0:
            idx = ((idx + 1) // 2) - 1      # go up to parent node
            idx_l = 2 * (idx + 1) - 1       # get its left child
            idx_r = 2 * (idx + 1)           # get its right child
            sp_l = self._sumtree[idx_l].summed_priority if self._sumtree[idx_l] else 0.
            sp_r = self._sumtree[idx_r].summed_priority if self._sumtree[idx_r] else 0.
            mp_l = self._sumtree[idx_l].max_priority if self._sumtree[idx_l] else 0.
            mp_r = self._sumtree[idx_r].max_priority if self._sumtree[idx_r] else 0.
            self._sumtree[idx] = PrioritizedExperienceTuple(
                experience_tuple=None,
                summed_priority=(sp_l + sp_r),
                max_priority=max(mp_l, mp_r))

    def _step(self):
        # steps the expiration index to the next leaf.
        experation_start_idx = self._tree_size - self._capacity
        memory_id = self._expiration_idx - experation_start_idx
        next_memory_id = (memory_id + 1) % self._capacity
        self._expiration_idx = experation_start_idx + next_memory_id
        self._total_steps += 1

    @property
    def num_items(self):
        return min(self._capacity, self._total_steps)

    def insert(self, experience_tuple):
        # inserts an experience tuple with max priority, updates upstream priorities,
        # and steps the expiration index.
        self._sumtree[self._expiration_idx] = PrioritizedExperienceTuple(
            experience_tuple=experience_tuple,
            summed_priority=self._sumtree[0].max_priority,
            max_priority=self._sumtree[0].max_priority)
        self._update_priorities(self._expiration_idx)
        self._step()

    def sample(self, batch_size):
        # samples a batch of experience tuples of size batch_size.
        p_total = self._sumtree[0].summed_priority

        # get uniform random numbers in intervals
        #     [0,p_tot/k), [p_tot/k,2*p_tot/k), [2*p_tot/k,3*p_tot/k), etc.
        shifts = np.arange(start=0, stop=batch_size, dtype=np.float32)
        shifts *= p_total / float(batch_size)
        unifs = np.random.uniform(low=0.0, high=1.0, size=batch_size)
        unifs *= p_total / float(batch_size)
        searches = shifts + unifs

        # marginal dist of a randomly selected search query has dist U[0, p_total).
        # now we search for memory_id in which each random query fell.
        indices = []
        tree_height = int(np.ceil(np.log(self._capacity) / np.log(2.0))) + 1
        for search in searches:
            sp_offset = 0.0
            idx = 0
            for _ in range(0, tree_height-1):
                idx_l = 2 * (idx + 1) - 1
                sp_l = self._sumtree[idx_l].summed_priority
                if search >= sp_offset + sp_l:
                    idx_r = 2 * (idx + 1)
                    idx = idx_r
                    sp_offset += sp_l
                else:
                    idx = idx_l
            indices.append(idx)

        data = [self._sumtree[i].experience_tuple for i in indices]
        weights = [self._sumtree[i].summed_priority ** (-self._beta) for i in indices]
        max_w = max(weights)
        weights = [w / max_w for w in weights]
        return {
            "indices": indices,
            "data": data,
            "weights": weights
        }

    def update_alpha(self, new_alpha):
        # updates priority exponent alpha.
        self._alpha = new_alpha

    def update_beta(self, new_beta):
        # updates importance weight exponent beta.
        self._beta = new_beta

    def update_td_errs(self, indices, td_errs):
        # updates td errors and associated priorities.
        for i, e in zip(indices, td_errs):
            pt = self._sumtree[i]
            pt.experience_tuple.td_err = min(max(-1.0, e), 1.0)

            priority = self._get_priority(pt.experience_tuple)
            pt.summed_priority = priority
            pt.max_priority = priority
            self._update_priorities(i)

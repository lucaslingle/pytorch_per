import numpy as np


class ExperienceTuple:
    def __init__(self, s_t, a_t, r_t, s_tp1, td_err):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = r_t
        self.s_tp1 = s_tp1
        self.td_err = td_err


class PrioritizedExperienceTuple:
    def __init__(self, priority, experience_tuple):
        self.priority = priority
        self.experience_tuple = experience_tuple


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

    def _get_capacity(self, capacity):
        # computes the number of leaf nodes to be used as memory.
        # we make the capacity a power of two to simplify the implementation.
        return 2 ** int(np.ceil(np.log(capacity) / np.log(2.0)))

    def _get_priority(self, experience_tuple):
        # computes priority using proportional prioritization.
        return (np.fabs(experience_tuple.td_err) + self._eps) ** self._alpha

    def _update_priorities(self, idx):
        # recomputes priorities for the nodes above index
        while idx != 0:
            idx = ((idx + 1) // 2) - 1      # go up to parent node
            idx_l = 2 * (idx + 1) - 1       # get its left child
            idx_r = 2 * (idx + 1)           # get its right child
            sp_l = self._sumtree[idx_l].priority if self._sumtree[idx_l] else 0.0
            sp_r = self._sumtree[idx_r].priority if self._sumtree[idx_r] else 0.0
            self._sumtree[idx] = PrioritizedExperienceTuple(
                priority=(sp_l + sp_r),
                experience_tuple=None)

    def _step(self):
        # steps the expiration index to the least recently written leaf index.
        experation_start_idx = self._tree_size - self._capacity
        memory_id = self._expiration_idx - experation_start_idx
        next_memory_id = (memory_id + 1) % self._capacity
        self._expiration_idx = experation_start_idx + next_memory_id
        self._total_steps += 1

    def insert(self, experience_tuple):
        # inserts an experience tuple, updates upstream priorities,
        # and steps the expiration index.
        priority = self._get_priority(experience_tuple)
        self._sumtree[self._expiration_idx] = PrioritizedExperienceTuple(
            priority=priority,
            experience_tuple=experience_tuple
        )
        self._update_priorities(self._expiration_idx)
        self._step()

    def sample(self, batch_size, debug=False):
        # samples a batch of experience tuples of size batch_size.
        assert self._total_steps >= self._capacity or debug
        p_total = self._sumtree[0].priority

        # get numbers in intervals
        #     [0,p_tot/k), [p_tot/k,2*p_tot/k), [2*p_tot/k,3*p_tot/k), etc.
        shifts = np.arange(start=0, stop=batch_size, dtype=np.float32)
        shifts *= p_total / float(batch_size)
        unifs = np.random.uniform(low=0.0, high=1.0, size=batch_size)
        unifs *= p_total / float(batch_size)
        searches = shifts + unifs

        # search for memory_id in which each random query fell
        # this is a form of stratified sampling; note that the marginal dist over samples
        # with the batch_idx marginalized out matches the p.m.f. defined by the priorities.
        idxs = []
        tree_height = int(np.ceil(np.log(self._capacity) / np.log(2.0))) + 1
        for search in searches:
            sp_offset = 0.0
            idx = 0
            for _ in range(0, tree_height-1):
                idx_l = 2 * (idx + 1) - 1
                sp_l = self._sumtree[idx_l].priority
                if search >= sp_offset + sp_l:
                    idx_r = 2 * (idx + 1)
                    idx = idx_r
                    sp_offset += sp_l
                else:
                    idx = idx_l
            idxs.append(idx)

        indices = idxs
        data = [self._sumtree[i].experience_tuple for i in idxs]
        weights = [self._sumtree[i].priority ** (-self._beta) for i in idxs]
        max_w = max(weights)
        weights = [w / max_w for w in weights]
        return {
            "indices": indices,
            "data": data,
            "weights": weights
        }

    def update_alpha(self, new_alpha):
        # update priority exponent alpha.
        self._alpha = new_alpha

    def update_beta(self, new_beta):
        # update importance-weight exponent beta.
        self._beta = new_beta

    def update_td_errs(self, idxs, td_errs):
        # update td errors and associated priorities.
        for i, e in zip(idxs, td_errs):
            pt = self._sumtree[i]
            pt.experience_tuple.td_err = e
            pt.priority = self._get_priority(pt.experience_tuple)
            self._update_priorities(i)

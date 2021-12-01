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
        # make capacity a power of two to simplify the implementation
        return 2 ** int(np.ceil(np.log(capacity) / np.log(2.0)))

    def _get_priority(self, experience_tuple):
        return (np.fabs(experience_tuple.td_err) + self._eps) ** self._alpha

    def _step(self):
        experation_start_idx = self._tree_size - self._capacity
        memory_id = self._expiration_idx - experation_start_idx
        next_memory_id = (memory_id + 1) % self._capacity
        self._expiration_idx = experation_start_idx + next_memory_id
        self._total_steps += 1

    def insert(self, experience_tuple):
        priority = self._get_priority(experience_tuple)
        priority_delta = priority
        if self._sumtree[self._expiration_idx] is not None:
            priority_delta -= self._sumtree[self._expiration_idx].priority

        self._sumtree[self._expiration_idx] = PrioritizedExperienceTuple(
            priority=priority,
            experience_tuple=experience_tuple
        )

        idx = self._expiration_idx
        while idx != 0:
            idx = ((idx + 1) // 2) - 1
            if self._sumtree[idx] is not None:
                self._sumtree[idx].priority += priority_delta
            else:
                idx_l = 2 * (idx + 1) - 1
                idx_r = 2 * (idx + 1)      
                sp_l = self._sumtree[idx_l].priority if self._sumtree[idx_l] else 0.0
                sp_r = self._sumtree[idx_r].priority if self._sumtree[idx_r] else 0.0
                self._sumtree[idx] = PrioritizedExperienceTuple(
                    priority=(sp_l + sp_r),
                    experience_tuple=None
                )
        self._step()

    def sample(self, batch_size, debug=False):
        assert self._total_steps >= self._capacity or debug
        p_total = self._sumtree[0].priority

        # uniform random numbers in range [0, p_total / k)
        unifs = np.random.uniform(low=0.0, high=1.0, size=batch_size)
        unifs *= p_total / float(batch_size)

        # separate shift for each uniform random number
        shifts = np.ones(dtype=np.float32, shape=(batch_size,))
        shifts *= p_total / float(batch_size)
        shifts = np.cumsum(shifts)
        shifts -= p_total / float(batch_size)

        # get numbers in intervals
        #     [0,p_tot/k], [p_tot/k,2*p_tot/k], [2*p_tot/k,3*p_tot/k], etc.
        searches = unifs + shifts

        # search for memory_id in which each random query fell
        # this is a form of stratified sampling; the marginal dist over samples
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
        self._alpha = new_alpha

    def update_beta(self, new_beta):
        self._beta = new_beta

    def update_td_errs(self, idxs, td_errs):
        for i, e in zip(idxs, td_errs):
            pt = self._sumtree[i]
            assert pt is not None
            p_old = pt.priority

            pt.experience_tuple.td_err = e
            p_new = self._get_priority(pt.experience_tuple)
            pt.priority = p_new

            priority_delta = p_new - p_old
            idx = i
            while idx != 0:
                idx = (idx + 1) // 2 - 1
                self._sumtree[idx].priority += priority_delta

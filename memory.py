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

        self._tree_size = self._capacity * (self._capacity + 1) // 2
        self._sumtree = [None for _ in range(self._tree_size)]
        self._expiration_idx = self._tree_size - self._capacity

    def _get_capacity(self, capacity):
        # make capacity a power of two to simplify the implementation
        return 2 ** np.ceil(np.log(capacity) / np.log(2.0))

    def _get_priority(self, experience_tuple: ExperienceTuple):
        return (np.fabs(experience_tuple.td_err) + self._eps) ** self._alpha

    def _step(self):
        experation_start_idx = self._tree_size - self._capacity
        memory_id = self._expiration_idx - experation_start_idx
        next_memory_id = (memory_id + 1) % self._capacity
        self._expiration_idx = experation_start_idx + next_memory_id
        self._total_steps += 1

    def insert(self, experience_tuple: ExperienceTuple):
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
            idx = ((idx + 1) // 2) - 1  # parent idx is at i//2 using base-1 indexing.
            if self._sumtree[idx] is not None:
                self._sumtree[idx].priority += priority_delta
            else:
                l_idx = 2 * (idx + 1) - 1  # left child of current node is at 2i using base-1 indexing.
                r_idx = 2 * (idx + 1)      # right child of current node is at 2i+1 using base-1 indexing.
                l_node = self._sumtree[l_idx]
                r_node = self._sumtree[r_idx]
                l_priority = 0.0 if l_node is None else l_node.priority
                r_priority = 0.0 if r_node is None else r_node.priority
                self._sumtree[idx] = PrioritizedExperienceTuple(
                    priority=(l_priority + r_priority),
                    experience_tuple=experience_tuple
                )
        self._step()

    def sample_batch(self, batch_size):
        assert self._total_steps is not None
        p_total = self._sumtree

        # uniform random numbers in range p_total / k
        unifs = np.random.uniform(low=0.0, hi=1.0, size=batch_size)
        unifs *= p_total
        unifs /= float(batch_size)

        # separate shift for each uniform random number
        shifts = np.ones(dtype=np.float32, shape=(batch_size,))
        shifts /= float(batch_size)
        shifts = np.cumsum(shifts)
        shifts -= (1.0 / float(batch_size))

        # get numbers in intervals
        #     [0,p_tot/k], [p_tot/k,2*p_tot/k], [2*p_tot/k,3*p_tot/k], etc.
        searches = unifs + shifts

        # search for memory_id in which each random query fell
        # this is a form of stratified sampling; the marginal dist over samples
        # with the batch_idx marginalized out matches the p.m.f. defined by the priorities.
        idxs = [0 for _ in range(batch_size)]
        for _ in range(0, np.ceil(np.log(self._capacity) / np.log(2.0))):
            go_right = [
                int(searches[i] > self._sumtree.priority[idxs[i]])
                for i in range(batch_size)
            ]
            idxs = [
                (1-go_right[i]) * (2*(idxs[i]+1)-1) + \
                go_right[i] * (2*(idxs[i]+1))
                for i in range(batch_size)
            ]

        return [
            self._sumtree[idxs[i]].experience_tuple
            for i in range(batch_size)
        ]

# something like this...

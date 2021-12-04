"""
Unit tests for replay.py
"""

import unittest

from per.algos.replay import (
    ExperienceTuple,
    PrioritizedReplayMemory,
)


class TestMemory(unittest.TestCase):
    def setUp(self):
        #      0
        #    1   2
        #   3 4 5 6
        # 78910 11121314
        self.mem = PrioritizedReplayMemory(
            capacity=8, alpha=1.0, beta=1.0, eps=0.01)
        self.et = ExperienceTuple(
            s_t=0, a_t=0, r_t=0.0, s_tp1=1, d_t=0.0, td_err=0.0)

    def test_capacity(self):
        mem = PrioritizedReplayMemory(
            capacity=3, alpha=1.0, beta=1.0, eps=0.0)
        self.assertEqual(mem._capacity, 4)

    def test_tree_size(self):
        self.assertEqual(self.mem._tree_size, 15)

    def test_step(self):
        self.assertEqual(self.mem._expiration_idx, 7)

        self.mem._step()
        self.assertEqual(self.mem._expiration_idx, 8)

        self.mem._expiration_idx = 14
        self.mem._step()
        self.assertEqual(self.mem._expiration_idx, 7)

    def test_insert(self):
        ### test write occurs at expiration index
        idx = self.mem._expiration_idx
        self.mem.insert(self.et)
        self.assertEqual(
            self.mem._sumtree[idx].experience_tuple, self.et)

        ### test that the priority is added to all parent nodes in the tree
        priority = self.mem._sumtree[idx].summed_priority
        parent_idxs = [3, 1, 0]
        for idx in parent_idxs:
            self.assertEqual(self.mem._sumtree[idx].summed_priority, priority)

        ### test that the priority is updated correctly after a tuple expires
        for i in range(1, self.mem._capacity):
            # add capacity-1 additional experience tuples
            et = ExperienceTuple(
                s_t=0, a_t=0, r_t=0.0, s_tp1=1, d_t=0.0, td_err=float(i))
            self.mem.insert(et)

        et = ExperienceTuple(
            s_t=0, a_t=0, r_t=0.0, s_tp1=1, d_t=0.0, td_err=float(self.mem._capacity))
        self.mem.insert(et)
        sps = [
            self.mem._sumtree[7].summed_priority + self.mem._sumtree[8].summed_priority,
            self.mem._sumtree[3].summed_priority + self.mem._sumtree[4].summed_priority,
            self.mem._sumtree[1].summed_priority + self.mem._sumtree[2].summed_priority
        ]
        for idx, sp in zip(parent_idxs, sps):
            self.assertEqual(self.mem._sumtree[idx].summed_priority, sp)

    def test_sample(self):
        self.mem.insert(self.et)
        results = self.mem.sample(batch_size=1)
        self.assertEqual(results['indices'][0], self.mem._tree_size-self.mem._capacity)
        self.assertEqual(results['data'][0], self.et)


if __name__ == '__main__':
    unittest.main()

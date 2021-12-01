"""
Unit tests for memory.py
"""

import unittest

from per.memory import (
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
            capacity=8, alpha=1.0, beta=1.0, eps=0.0)
        self.et = ExperienceTuple(
            s_t=0, a_t=0, r_t=0.0, s_tp1=1, td_err=1.0)

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
        idx = self.mem._expiration_idx
        self.mem.insert(self.et)
        self.assertEqual(
            self.mem._sumtree[idx].experience_tuple, self.et)

        priority = self.mem._sumtree[idx].priority
        while idx != 0:
            idx = (idx+1) // 2 - 1
            self.assertEqual(
                self.mem._sumtree[idx].priority, priority)

    def test_sample_batch(self):
        self.mem.insert(self.et)
        results = self.mem.sample_batch(batch_size=1, debug=True)
        self.assertEqual(results['indices'][0], self.mem._tree_size-self.mem._capacity)
        self.assertEqual(results['data'][0], self.et)


if __name__ == '__main__':
    unittest.main()

"""
Unit tests for memory.py
"""

import unittest

from per.memory import (
    ExperienceTuple,
    PrioritizedReplayMemory,
)


class TestMemory(unittest.TestCase):
    def test_capacity(self):
        mem = PrioritizedReplayMemory(
            capacity=3, alpha=1.0, beta=1.0, eps=0.0)
        self.assertEqual(mem._capacity, 4)

    def test_tree_size(self):
        #      0
        #    1   2
        #   3 4 5 6
        # 78910 11121314
        mem = PrioritizedReplayMemory(
            capacity=8, alpha=1.0, beta=1.0, eps=0.0)
        self.assertEqual(mem._tree_size, 15)

    def test_step(self):
        mem = PrioritizedReplayMemory(
            capacity=8, alpha=1.0, beta=1.0, eps=0.0)
        self.assertEqual(mem._expiration_idx, 7)
        mem._step()
        self.assertEqual(mem._expiration_idx, 8)
        mem._expiration_idx = 14
        mem._step()
        self.assertEqual(mem._expiration_idx, 7)

    def test_insert(self):
        mem = PrioritizedReplayMemory(
            capacity=8, alpha=1.0, beta=1.0, eps=0.0)
        et = ExperienceTuple(
            s_t=0, a_t=0, r_t=0.0, s_tp1=1, td_err=1.0)
        idx = mem._expiration_idx
        mem.insert(et)
        self.assertEqual(
            mem._sumtree[idx].experience_tuple, et)

        priority = mem._sumtree[idx].priority
        while idx != 0:
            idx = (idx+1) // 2 - 1
            self.assertEqual(
                mem._sumtree[idx].priority, priority)

    def test_sample_batch(self):
        mem = PrioritizedReplayMemory(
            capacity=8, alpha=1.0, beta=1.0, eps=0.0)
        et = ExperienceTuple(
            s_t=0, a_t=0, r_t=0.0, s_tp1=1, td_err=1.0)
        mem.insert(et)
        results = mem.sample_batch(batch_size=1, debug=True)
        self.assertEqual(results[0][0], mem._tree_size-mem._capacity)
        self.assertEqual(results[0][1], et)


if __name__ == '__main__':
    unittest.main()

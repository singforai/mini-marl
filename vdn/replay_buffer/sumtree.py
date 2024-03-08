import numpy as np
import random

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children


class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.priority_tree = np.zeros(2 * self.capacity - 1)
        self.buffer = np.zeros(self.capacity, dtype=list)
        self.n_data = 0
        self.n_buffer = 0

    def _propagate(self, idx, change):
        parent_node = (idx - 1) // 2

        self.priority_tree[parent_node] += change

        if parent_node != 0:
            self._propagate(parent_node, change)

    def _retrieve_max(self, idx, segment_index):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.priority_tree):
            return idx

        if segment_index <= self.priority_tree[left]:
            return self._retrieve_max(left, segment_index)
        else:
            return self._retrieve_max(right, segment_index - self.priority_tree[left])

    def total(self):
        return self.priority_tree[0]

    def add(self, priority, sample):
        priority_idx = self.n_data + self.capacity - 1

        if self.n_data < self.capacity:
            priority_idx = self.n_data + self.capacity - 1
            self.buffer[self.n_data] = sample
        else:
            index: int = 0
            while True:
                priority_idx = np.argsort(self.priority_tree)[index]
                if priority_idx >= self.capacity-1:
                    break
                else:
                    index += 1

            self.buffer[priority_idx - self.capacity + 1] = sample

        self.update(priority_idx, priority)

        if self.n_data < self.capacity:
            self.n_data += 1

        return self.n_data

    def update(self, priority_idx, priority):
        change = priority - self.priority_tree[priority_idx]
        self.priority_tree[priority_idx] = priority
        self._propagate(priority_idx, change)
        # self.priority_array = self.priority_tree[self.capacity-1:]

    # get priority and sample
    def get(self, segment_index):
        idx = self._retrieve_max(0, segment_index)
        dataIdx = idx - self.capacity + 1
        return (idx, self.priority_tree[idx], dataIdx)

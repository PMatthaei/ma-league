import random

import numpy as np

from marl.components import EpisodeBatch


class BinarySumTree:
    write = 0

    def __init__(self, capacity):
        """
        A binary tree data structure where the parent’s value is the sum of its children
        :param capacity:
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        Update to the root node
        :param idx:
        :param change:
        :return:
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Find sample on leaf node
        :param idx:
        :param s:
        :return:
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p):
        """
        Store priority and sample
        :param p:
        :param data:
        :return:
        """
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Get priority and sample
        :param s:
        :return:
        """
        idx = self._retrieve(0, s)
        didx = idx - self.capacity + 1

        return idx, self.tree[idx], didx


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = BinarySumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class PrioritizedReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        """
        ReplayBuffer is a EpisodeBatch which is caped in size by its buffer size.
        :param scheme:
        :param groups:
        :param buffer_size:
        :param max_seq_length:
        :param preprocess:
        :param device:
        """
        super(PrioritizedReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length,
                                                      preprocess=preprocess,
                                                      device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.tree = BinarySumTree(capacity=buffer_size)
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def insert_episode_batch(self, ep_batch: EpisodeBatch, errors=None):
        # If buffer does not overflow with new episode batch
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            priorities  = self._get_priority(errors)
            self.tree.add(priorities, ep_batch)
            # Add transition data as samples to buffer
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            # Add episode data as samples to buffer
            self.update(ep_batch.data.episode_data, slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        # If buffer overflows with new episode batch
        else:
            # Slice episode batch to size (buffer_left) fitting into buffer and insert recursively
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size) -> EpisodeBatch:
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]  # return complete buffer if the buffer is filled with one batch
        else:
            # Uniform sampling - Choose episode ids to include into the sample
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def sample(self, batch_size: int) -> EpisodeBatch:
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]  # return complete buffer if the buffer is filled with one batch
        else:
            batch = []
            ep_ids = []
            segment = self.tree.total() / batch_size
            priorities = []

            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)

                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                priorities.append(p)
                batch.append(data)
                ep_ids.append(idx)

            sampling_probabilities = priorities / self.tree.total()
            is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
            is_weight /= is_weight.max()

            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())

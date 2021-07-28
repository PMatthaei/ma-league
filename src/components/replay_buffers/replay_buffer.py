from components.episode_batch import EpisodeBatch
import numpy as np


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size: int, max_seq_length: int, preprocess=None, device="cpu"):
        """
        ReplayBuffer is a EpisodeBatch which is caped in size by its buffer size.
        :param scheme:
        :param groups:
        :param buffer_size:
        :param max_seq_length:
        :param preprocess:
        :param device:
        """
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess,
                                           device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch: EpisodeBatch):
        # If buffer does not overflow with new episode batch
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
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

    def can_sample(self, batch_size: int) ->  bool:
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size: int) -> EpisodeBatch:
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]  # return complete buffer if the buffer is filled with one batch
        else:
            # Uniform sampling - Choose episode ids to include into the sample
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())

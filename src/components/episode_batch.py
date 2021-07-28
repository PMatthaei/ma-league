from types import SimpleNamespace as SN

import numpy as np
import torch as th


def _check_safe_view(v, dest, key):
    idx = len(v.shape) - 1
    for s in dest.shape[::-1]:
        if v.shape[idx] != s:
            if s != 1:
                raise ValueError("Unsafe reshape of {} to {} at Key: {}".format(v.shape, dest.shape, key))
        else:
            idx -= 1


def _get_num_items(indexing_item, max_size):
    # Get the number of items depending on the type of indexing_item
    if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
        return len(indexing_item)
    elif isinstance(indexing_item, slice):
        _range = indexing_item.indices(max_size)
        return 1 + (_range[1] - _range[0] - 1) // _range[2]


def _new_data_sn():
    # Returns empty batch data
    new_data = SN()
    new_data.transition_data = {}
    new_data.episode_data = {}  # ! This is only used if episodes are of constant length !
    return new_data


def _parse_slices(items):
    parsed = []
    # Only batch slice given, add full time slice
    if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
    ):
        items = (items, slice(None))

    # Need the time indexing to be contiguous
    if isinstance(items[1], list):
        raise IndexError("Indexing across Time must be contiguous")

    for item in items:
        # TODO: stronger checks to ensure only supported options get through
        if isinstance(item, int):
            # Convert single indices to slices
            parsed.append(slice(item, item + 1))
        else:
            # Leave slices and lists as is
            parsed.append(item)
    return parsed


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        """
        Batch of episodes.
        :param scheme:
        :param groups:
        :param batch_size: Amount of episodes within this batch.
        :param max_seq_length: Longest episode
        :param data: Data of all episodes
        :param preprocess: Pre-processing steps to the data
        :param device: Device-type for tensors
        """
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = _new_data_sn()
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        # Preprocess scheme
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]  # name of the preprocessing f.e. "actions_onehot"
                transforms = preprocess[k][1]  # transform function

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                # Process all transforms in execution order provided
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # Apply scheme changes
                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                # Carry group key over to new scheme value f.e. for "actions"
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                # Carry episode_const key over to new scheme value
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        # "filled" not allowed as key since used for masking in learners -> add to scheme
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        # Setup scheme
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)  # episode_const is per default False
            group = field_info.get("group", None)  # group is per default None
            dtype = field_info.get("dtype", th.float32)  # dtype is per default th.float32

            # Transform ints to tuple for later tensor construction
            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:  # If group is set f.e. in "actions" to "agents", key "agents must occur in the groups scheme
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)  # fetch group shape for desired key in groups scheme
            else:
                shape = vshape

            if episode_const:  # always same episode length -> leave out max_seq_length in tensor construction
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype,
                                                                device=self.device)

    def extend(self, scheme, groups=None):
        # Extend via new setup -> groups are carried over if set to None (default)
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        # Convert all data from the scheme to specified device
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = _parse_slices((bs, ts))
        for key, value in data.items():
            # Find target of the update process -> transition data
            if key in self.data.transition_data:
                target = self.data.transition_data
                # Marked filled portions of the slices provided with 1
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            # Find target of the update process -> episode data
            elif key in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(key))

            # Get data type and value of the scheme data
            dtype = self.scheme[key].get("dtype", th.float32)
            if isinstance(value, list):
                value = th.tensor(value, dtype=dtype, device=self.device)
            else:
                value = value.to(dtype).to(device=self.device)

            # Check validity of following view_as
            _check_safe_view(value, target[key][_slices], key)
            # Add value via view_as
            target[key][_slices] = value.view_as(target[key][_slices])

            # Perform pre-processing
            if key in self.preprocess:
                new_k = self.preprocess[key][0]  # Get new key defined by preprocess method
                value = target[key][_slices]  # Get original value
                for transform in self.preprocess[key][1]:  # Get all transforms and apply them in array order
                    value = transform.transform(value)
                # Add transformed value via view_as
                _check_safe_view(value, target[new_k][_slices], key)
                target[new_k][_slices] = value.view_as(target[new_k][_slices])

    def __getitem__(self, item):
        # If item is a string
        if isinstance(item, str):
            # implement [""] to get item from batch data
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        # If item is a string only tuple
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = _new_data_sn()
            # Copy all values from keys into new_data
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data,
                               device=self.device)
            return ret
        else:
            item = _parse_slices(item)
            new_data = _new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = _get_num_items(item[0], self.batch_size)
            ret_max_t = _get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def max_t_filled(self):
        # Return the number of the maximum timestep until which a episode is filled (episodes have different length)
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())

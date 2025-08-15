import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset


class InfSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch = 0
        self.reset_sampler()

    def reset_sampler(self):
        num = len(self.dataset)
        if self.shuffle:
            # 使用当前 epoch 作为随机种子的一部分，确保每个 epoch 的打乱是不同但确定的
            g = torch.Generator()
            g.manual_seed(torch.initial_seed() + self.epoch)
            indices = torch.randperm(num, generator=g)
        else:
            indices = torch.arange(num)
        self.indices = indices.tolist()
        self.iter_num = 0
        self.epoch += 1

    def __iter__(self):
        return self

    def __next__(self):
        value = self.indices[self.iter_num]
        self.iter_num = self.iter_num + 1

        if self.iter_num >= len(self.indices):
            self.reset_sampler()
        return value

    def __len__(self):
        return len(self.dataset)


class DistributedInfSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        super().__init__(dataset, shuffle=shuffle)
        self.epoch = 0
        self.reset_sampler()

    def reset_sampler(self):
        # 设置当前 epoch，这会影响 DistributedSampler 的随机打乱
        super().set_epoch(self.epoch)
        self.indices = list(super().__iter__())
        self.iter_num = 0
        self.epoch += 1

    def __iter__(self):
        return self

    def __next__(self):
        value = self.indices[self.iter_num]
        self.iter_num = self.iter_num + 1

        if self.iter_num >= len(self.indices):
            self.reset_sampler()
        return value

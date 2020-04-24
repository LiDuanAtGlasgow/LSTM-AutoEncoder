import torch
from torch.utils.data.sampler import Sampler

class BatchRandomSampler(Sampler):
    def __init__(self,indices,bacth_size):
        self.indices=indices
        self.bactch_size=bacth_size
    def __iter__(self):
        bactch_size=self.bactch_size
        indices=self.indices
        iter_number=int(len(indices)/bactch_size)
        random_matrix=torch.randperm(iter_number)
        for i in range(iter_number):
            t=random_matrix[i]
            random_indices=indices[t*bactch_size:(t+1)*bactch_size]
            indices[i*bactch_size:(i+1)*bactch_size]=random_indices
        return iter(indices)
    def __len__(self):
        return len(self.indices)        


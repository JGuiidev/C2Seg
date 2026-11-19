"""Full global contrastive batches with unique image indices within each batch."""
from __future__ import annotations
import math
import torch
from torch.utils.data import Sampler


class FullContrastiveBatchSampler(Sampler):
    def __init__(self,n,batch_size=256,seed=42):
        if n<batch_size or batch_size<2:
            raise ValueError('Dataset must have at least batch_size distinct images')
        self.n=int(n);self.batch_size=int(batch_size);self.seed=int(seed);self.epoch=0

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __len__(self):return math.ceil(self.n/self.batch_size)

    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed+self.epoch*7919)
        shuffled=torch.randperm(self.n,generator=gen).tolist()
        missing=len(self)*self.batch_size-self.n
        # Take repeated images from a fresh shuffle; only duplicate across batches.
        tail=torch.randperm(self.n,generator=gen)[:missing].tolist()
        full=shuffled+tail
        for off in range(0,len(full),self.batch_size):
            group=full[off:off+self.batch_size]
            if len(set(group))!=len(group):
                # Reposition repeats away from earlier copies in the final batch.
                used=set(group)
                for j,item in enumerate(group):
                    if group.count(item)>1:
                        alt=next(idx for idx in range(self.n) if idx not in used)
                        group[j]=alt;used.add(alt)
            if len(group)!=self.batch_size or len(set(group))!=len(group):
                raise RuntimeError('Stage I batch uniqueness failure')
            yield group

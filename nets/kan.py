"""
Compact spline-based Kolmogorov–Arnold layer shared by encoder, BCAM and K-Gate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-3.,3.), chunk_size=4096, activation_checkpoint=True):
        super().__init__()
        if grid_size < 2 or spline_order < 1 or grid_range[1]<=grid_range[0]:
            raise ValueError('Invalid KAN grid configuration')
        self.in_features=int(in_features); self.out_features=int(out_features)
        self.grid_size=int(grid_size);self.spline_order=int(spline_order)
        self.chunk_size=int(chunk_size)
        self.activation_checkpoint=bool(activation_checkpoint)
        h=(grid_range[1]-grid_range[0])/grid_size
        knots=torch.arange(-spline_order,grid_size+spline_order+1,dtype=torch.float32)*h+grid_range[0]
        self.register_buffer('grid',knots)
        self.base_weight=nn.Parameter(torch.empty(out_features,in_features))
        self.spline_weight=nn.Parameter(torch.empty(out_features,in_features,grid_size+spline_order))
        self.spline_scaler=nn.Parameter(torch.empty(out_features,in_features))
        nn.init.kaiming_uniform_(self.base_weight,a=math.sqrt(5))
        nn.init.normal_(self.spline_weight,std=0.01)
        nn.init.constant_(self.spline_scaler,0.2)

    def b_splines(self,x):
        if x.ndim!=2 or x.shape[1]!=self.in_features:raise ValueError('KAN input dimension mismatch')
        knots=self.grid
        v=x.unsqueeze(-1)
        bases=((v>=knots[:-1]) & (v<knots[1:])).to(v.dtype)
        for order in range(1,self.spline_order+1):
            left=(v-knots[:-(order+1)])/(knots[order:-1]-knots[:-(order+1)])
            right=(knots[order+1:]-v)/(knots[order+1:]-knots[1:-order])
            bases=left*bases[...,:-1]+right*bases[...,1:]
        return bases

    def _project(self,x):
        base=F.linear(F.silu(x),self.base_weight.float())
        bases=self.b_splines(x)
        weights=self.spline_weight.float()*self.spline_scaler.float().unsqueeze(-1)
        spline=F.linear(bases.reshape(x.shape[0],-1),weights.reshape(self.out_features,-1))
        return base+spline

    def forward(self,x):
        """Exact KAN projection with optional recomputation of spline activations.

        Chunking bounds each spline workspace; non-reentrant activation checkpointing
        avoids retaining its large intermediate bases across the entire backward pass.
        It does not split a training batch or modify the optimizer update frequency.
        """
        shape=x.shape
        if shape[-1]!=self.in_features:
            raise ValueError('KAN last dimension mismatch')
        # Spline recursion remains FP32 for numerical stability, even under AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            flat=x.reshape(-1,self.in_features).float()
            chunks=flat.split(self.chunk_size) if self.chunk_size>0 else (flat,)
            outputs=[]
            for chunk in chunks:
                if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                    # The projection has no RNG-dependent operations. Non-reentrant
                    # checkpoint also handles inputs which do not require gradients.
                    y=checkpoint(self._project,chunk,use_reentrant=False,preserve_rng_state=False)
                else:
                    y=self._project(chunk)
                outputs.append(y)
            out=outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=0)
        return out.reshape(*shape[:-1],self.out_features).to(x.dtype)

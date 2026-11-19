"""Train-only CLIP semantics, exact-text positives, and sparse trusted neighbours."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def calibrate_text_bank(embeddings, alpha=0.7, center=None):
    """Remove the shared direction estimated from unique TRAIN text vectors."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    base = F.normalize(embeddings.float(), dim=-1)
    mean_direction = base.mean(0, keepdim=True) if center is None else center.float().to(base.device)
    centred = F.normalize(base - mean_direction, dim=-1)
    return F.normalize((1-alpha)*base + alpha*centred, dim=-1)


class SemanticTeacher:
    """Frozen text bank: unique-text neighbours defined using only training data.

    Mutual top-k neighbours must exceed each row's median + IQR * margin_factor.
    Targets adapt semantic mass to the confidence gap; exact texts are positives.
    """
    def __init__(self, text_bank, alpha=.7, top_k=4, rho_max=.3, margin_factor=.5, temperature=.07, center=None):
        if top_k<1 or not 0<=rho_max<=1 or temperature<=0 or margin_factor<0:
            raise ValueError('Invalid semantic teacher settings')
        self.bank=calibrate_text_bank(text_bank,alpha,center=center).detach()
        unique=len(self.bank)
        sim=self.bank@self.bank.T
        sim.fill_diagonal_(-float('inf'))
        k=min(int(top_k),max(0,unique-1))
        neighbours=torch.zeros((unique,unique),dtype=torch.bool,device=self.bank.device)
        quality=torch.zeros_like(sim)
        if k:
            top=sim.topk(k,dim=1).indices
            neighbours.scatter_(1,top,True)
            neighbours=neighbours & neighbours.T
            # Quantile over valid distinct descriptions; finite values only.
            ordered=sim.sort(dim=1).values[:,1:]  # drop diagonal -inf, keep highest distinct-text match
            q25=torch.quantile(ordered,.25,dim=1)
            q50=torch.quantile(ordered,.5,dim=1)
            q75=torch.quantile(ordered,.75,dim=1)
            cutoff=q50+margin_factor*(q75-q25).clamp_min(1.e-5)
            confidence=((sim-cutoff[:,None])/(q75-q25).clamp_min(1.e-5)[:,None]).clamp_min(0)
            neighbours=neighbours & (confidence>0)
            quality=confidence.masked_fill(~neighbours,0).clamp(max=10)
        self.quality=quality
        self.rho_max=float(rho_max)
        self.temperature=float(temperature)
        self.top_k=int(top_k)

    def targets(self, ids):
        ids=ids.to(device=self.bank.device,dtype=torch.long).view(-1)
        if ids.numel()<2:
            raise ValueError('Contrastive batch must contain >= 2 samples')
        if (ids<0).any() or (ids>=len(self.bank)).any():
            raise ValueError('Text ID outside semantic teacher bank')
        positive=(ids[:,None]==ids[None,:]).float()
        ppos=positive/positive.sum(1,keepdim=True).clamp_min(1)
        q=self.quality[ids[:,None],ids[None,:]]
        q=q.masked_fill(positive.bool(),0)
        # Missing neighbours do not dilute exact-text supervision.
        confidence=q.sum(1)
        has=confidence>0
        psem=torch.where(has[:,None],q/confidence[:,None].clamp_min(1e-8),torch.zeros_like(q))
        rho=(self.rho_max*(1-torch.exp(-confidence))).clamp(0,self.rho_max)
        it=(1-rho[:,None])*ppos+rho[:,None]*psem
        it=it/it.sum(1,keepdim=True).clamp_min(1.e-8)
        # Correct conditional targets: each text query is normalized across images.
        ti=it.T/it.T.sum(1,keepdim=True).clamp_min(1.e-8)
        return it,ti

    def embeddings(self,ids):
        return self.bank[ids.to(self.bank.device,dtype=torch.long)]

"""Multi-positive global / foreground alignment and foreground-background separation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.semantic_targets import SemanticTeacher


class Stage1Heads(nn.Module):
    def __init__(self,dim=512):
        super().__init__()
        self.global3=nn.Linear(256,dim)
        self.global4=nn.Linear(512,dim)
        self.region2=nn.Linear(128,dim)
        self.region3=nn.Linear(256,dim)

    def forward(self,features):
        f2,f3,f4=features[2:5]
        return self.global3(f3.float().mean((-2,-1))),self.global4(f4.float().mean((-2,-1)))


def region_pool(feature,mask):
    """Area-preserving mask resize (224->56/28), stable for tiny lesions.

    Mask has no gradient. AvgPool with integer factor avoids nondeterministic
    adaptive-average-pool CUDA backward and preserves fractional occupancy.
    """
    h,w=feature.shape[-2:]
    hh,ww=mask.shape[-2:]
    if hh%h or ww%w:
        raise ValueError('Mask must be divisible by feature spatial shape')
    m=F.avg_pool2d(mask.float(),kernel_size=(hh//h,ww//w),stride=(hh//h,ww//w))
    feat=feature.float()
    fg_sum=m.sum((2,3))
    bg_sum=(1-m).sum((2,3))
    fg=(feat*m).sum((2,3))/fg_sum.clamp_min(1e-6)
    bg=(feat*(1-m)).sum((2,3))/bg_sum.clamp_min(1e-6)
    valid=(fg_sum.squeeze(1)>1e-6)&(bg_sum.squeeze(1)>1e-6)
    return fg,bg,valid


def soft_target_ce(visual,text,ids,teacher,temperature=.07):
    if len(ids)<2:
        return visual.sum()*0
    with torch.no_grad():
        target_it,target_ti=teacher.targets(ids)
    logits=F.normalize(visual.float(),dim=-1)@F.normalize(text.float(),dim=-1).T/temperature
    it=-(target_it*F.log_softmax(logits,dim=1)).sum(1).mean()
    ti=-(target_ti*F.log_softmax(logits.T,dim=1)).sum(1).mean()
    # KL removes the irreducible target entropy, unlike raw CE.
    entropy_it=-(target_it*target_it.clamp_min(1e-12).log()).sum(1).mean()
    entropy_ti=-(target_ti*target_ti.clamp_min(1e-12).log()).sum(1).mean()
    return (it+ti-entropy_it-entropy_ti)*.5


def stage1_objective(features,masks,ids,teacher,heads,cfg):
    text=teacher.embeddings(ids)
    z3,z4=heads(features)
    zero=z4.sum()*0.0
    global_loss=soft_target_ce(z4,text,ids,teacher,cfg.stage1_tau)
    if cfg.stage1_multiscale_global:
        global_loss=global_loss*.7+soft_target_ce(z3,text,ids,teacher,cfg.stage1_tau)*.3
    region_loss=zero
    sep_loss=zero
    margins=[]
    if cfg.stage1_use_region or cfg.stage1_use_separation:
        scales=(2,3) if cfg.stage1_region_scales=='23' else (3,)
        losses=[]
        separations=[]
        for s in scales:
            fg,bg,valid=region_pool(features[s],masks)
            head=heads.region2 if s==2 else heads.region3
            if valid.any():
                fg_emb=head(fg[valid]); bg_emb=head(bg[valid]); txt=text[valid]; selected=ids[valid]
                if cfg.stage1_use_region:
                    losses.append((.6 if s==2 else .4,soft_target_ce(fg_emb,txt,selected,teacher,cfg.stage1_tau)))
                if cfg.stage1_use_separation:
                    f=F.normalize(fg_emb.float(),dim=-1)
                    b=F.normalize(bg_emb.float(),dim=-1)
                    margin=(f*txt).sum(-1)-(b*txt).sum(-1)
                    margins.append(margin.detach().mean())
                    separations.append((.6 if s==2 else .4,F.softplus((cfg.stage1_separation_margin-margin)/cfg.stage1_region_tau).mean()))
        if losses:
            region_loss=sum(w*term for w,term in losses)/sum(w for w,_ in losses)
        if separations:
            sep_loss=sum(w*term for w,term in separations)/sum(w for w,_ in separations)
    terms=[]
    terms.append((float(cfg.stage1_global_weight),global_loss))
    if cfg.stage1_use_region:terms.append((float(cfg.stage1_region_weight),region_loss))
    if cfg.stage1_use_separation:terms.append((float(cfg.stage1_separation_weight),sep_loss))
    total=sum(w*v for w,v in terms)/sum(w for w,_ in terms)
    diagnostic={'global':float(global_loss.detach()),'region':float(region_loss.detach()),
                'separation':float(sep_loss.detach()),
                'margin':float(torch.stack(margins).mean()) if margins else 0.0}
    return total,diagnostic

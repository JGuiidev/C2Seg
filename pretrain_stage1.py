"""Stage I multi-positive semantic pretraining.

One genuine optimizer step per 256 training images.
"""
from __future__ import annotations
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from args import run_paths
from datasets.dataset import MedicalDataset,worker_seed
from datasets.stage1_sampler import FullContrastiveBatchSampler
from tools.log_utils import create_logger
from tools.train_utils import amp_mode,seed_everything,device_for
from tools.semantic_targets import SemanticTeacher
from tools.stage1_losses import Stage1Heads,stage1_objective


STAGE1_VERSION='C2SEG_STAGE1_V1'
# Compare these training settings directly when reusing Stage I checkpoints.
STAGE1_FIELDS=(
 'dataset','data_root','clip_weights_path','clip_backbone','pair_rule','image_size','text_len',
 'preprocess','image_channel_order','image_normalization','train_mask_interpolation',
 'evaluation_protocol','kan_grid_size','kan_spline_order','kan_range','kan_chunk_size',
 'kan_residual_scale','kan_activation_checkpoint','seed','deterministic','amp','num_workers',
 'stage1_epochs','stage1_train_epochs','stage1_checkpoint_epochs','stage1_batch_size','stage1_lr_cnn','stage1_lr_kan','stage1_lr_head',
 'stage1_weight_decay','stage1_warmup_epochs','stage1_eta_min','stage1_tau',
 'stage1_alpha','stage1_top_k','stage1_rho_max','stage1_neighbor_margin',
 'stage1_multiscale_global','stage1_use_neighbors','stage1_use_region',
 'stage1_use_separation','stage1_region_scales','stage1_global_weight',
 'stage1_region_weight','stage1_separation_weight','stage1_separation_margin',
 'stage1_region_tau','stage1_photometric_aug',
)


def _atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _torch_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    try:
        torch.save(obj,tmp);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def _load(path):
    try:return torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:return torch.load(path,map_location='cpu')


def stage1_settings(config):
    """Record plain Stage I settings for transparent cache matching."""
    return {name: getattr(config, name) for name in STAGE1_FIELDS}


def _stage1_paths(config):
    result,weight=run_paths(config)
    if result.name!='stage2' or weight.name!='stage2':
        raise ValueError('Stage I requires a run_name ending in /stage2')
    stage_result=result.parent/'stage1'
    stage_weight=weight.parent/'stage1'
    return stage_result,stage_weight


def _load_encoder(path,encoder,epoch):
    payload=_load(path)
    if payload.get('version') != STAGE1_VERSION or int(payload.get('epoch', -1)) != int(epoch):
        raise RuntimeError('Stage I checkpoint format or epoch mismatch')
    saved=payload['encoder']
    current=encoder.state_dict()
    if set(current) != set(saved):
        raise RuntimeError('Stage I encoder keys differ unexpectedly')
    for key,tensor in saved.items():
        if current[key].shape!=tensor.shape:
            raise RuntimeError(f'Stage I encoder shape mismatch: {key}')
    current.update(saved)
    encoder.load_state_dict(current,strict=True)
    return payload


def _make_stage1_loaders(config):
    cfg=SimpleNamespace(**vars(config))
    cfg.geometry_aug='none'
    cfg.use_photometric_aug=bool(config.stage1_photometric_aug)
    cfg.use_blur_aug=False
    ds=MedicalDataset(cfg.data_root,'Train',cfg)
    val_ds=MedicalDataset(cfg.data_root,'Val',cfg)
    sampler=FullContrastiveBatchSampler(len(ds),int(config.stage1_batch_size),int(config.seed))
    # Worker seeds independent from Stage II loader and consistent between experiments.
    def worker_kwargs(seed):
        return dict(num_workers=int(config.num_workers),pin_memory=True,
                    worker_init_fn=worker_seed,
                    persistent_workers=config.num_workers>0,
                    generator=torch.Generator().manual_seed(seed))
    training=DataLoader(ds,batch_sampler=sampler,**worker_kwargs(config.seed+601))
    validation=DataLoader(val_ds,batch_size=int(config.stage1_batch_size),shuffle=False,
                          **worker_kwargs(config.seed+602))
    return training,validation,sampler


@torch.inference_mode()
def _text_bank(clip,ds,device):
    outputs=[]
    for batch in ds.unique_tokens.split(128):
        tokens=batch.to(device)
        mask=tokens.ne(0)
        result=clip.encode_text(tokens,mask=mask)
        if isinstance(result,tuple):result=result[0]
        outputs.append(result.float())
    return torch.nn.functional.normalize(torch.cat(outputs,dim=0),dim=-1)


def _teacher_for_split(bank,config,center=None):
    return SemanticTeacher(bank,alpha=config.stage1_alpha,
        top_k=config.stage1_top_k if config.stage1_use_neighbors else 1,
        rho_max=config.stage1_rho_max if config.stage1_use_neighbors else 0.,
        margin_factor=config.stage1_neighbor_margin,
        temperature=config.stage1_tau,center=center)


def _cache_paths(config):
    result, weight = run_paths(config)
    # One human-readable cache per dataset, checked against its config.json.
    return result.parent.parent / '_stage1_cache', weight.parent.parent / '_stage1_cache'


def stage1_lr_factor(epoch,config):
    warmup=int(config.stage1_warmup_epochs)
    horizon=int(config.stage1_epochs)
    if epoch<=warmup:return epoch/max(1,warmup)
    progress=min(1.,(epoch-warmup)/max(1,horizon-warmup))
    return .5*(1+math.cos(math.pi*progress))


def run_stage1(config,encoder,clip,logger=None):
    """Train or reuse Stage I and transfer the selected encoder weights."""
    if int(config.stage1_batch_size)!=256:
        raise ValueError('Stage I batch size must equal 256')
    stage_result,stage_weight=_stage1_paths(config)
    cache_result,cache_weight=_cache_paths(config)
    requested_settings=stage1_settings(config)
    epoch_selected=int(config.stage1_checkpoint_epoch)
    selected_name=f'epoch_{epoch_selected:03d}.pth'
    selected_cache=cache_weight/selected_name
    local_ckpt=stage_weight/'selected.pth'
    meta_file=stage_result/'metrics.json'
    cache_meta=cache_result/'metrics.json'
    cache_epochs=[int(n) for n in config.stage1_checkpoint_epochs]

    def use_cache(meta):
        if meta.get('status') != 'completed':
            return None
        stored = json.loads((cache_result / 'config.json').read_text(encoding='utf-8'))
        if stored != requested_settings:
            return None
        if not all((cache_weight/f'epoch_{i:03d}.pth').is_file() for i in cache_epochs):
            return None
        payload=_load_encoder(selected_cache,encoder,epoch_selected)
        stage_result.mkdir(parents=True,exist_ok=True)
        stage_weight.mkdir(parents=True,exist_ok=True)
        _atomic_json(stage_result/'config.json', {'selected_epoch':epoch_selected, 'stage1':requested_settings})
        shutil.copy2(selected_cache,local_ckpt)
        (stage_result/'train.log').write_text(f'Stage I reused epoch {epoch_selected} from {selected_cache}\n',encoding='utf-8')
        local={**meta,'selected_epoch':epoch_selected,'checkpoint':str(local_ckpt),
               'reused':True,'training_seconds':0.0,
               'cache_training_seconds':meta.get('training_seconds',0.0),
               'selected_val_loss':meta.get('val_loss_by_epoch',{}).get(str(epoch_selected))}
        _atomic_json(meta_file,local)
        if logger:logger.info('Stage I reused cached epoch E%d',epoch_selected)
        return {'checkpoint':str(local_ckpt),
                'selected_epoch':epoch_selected,'reused':True,'training_seconds':0.0}

    if cache_meta.is_file():
        try:
            previous=json.loads(cache_meta.read_text(encoding='utf-8'))
            ready=use_cache(previous)
            if ready:return ready
        except (ValueError,RuntimeError,KeyError,EOFError,OSError) as exc:
            if logger:logger.warning('Stage I cache rejected: %s',exc)

    stage_result.mkdir(parents=True,exist_ok=True)
    stage_weight.mkdir(parents=True,exist_ok=True)
    cache_result.mkdir(parents=True,exist_ok=True)
    cache_weight.mkdir(parents=True,exist_ok=True)
    # Never classify partial/corrupt runs as valid cache entries.
    cache_meta.unlink(missing_ok=True)
    for file in cache_weight.glob('epoch_*.pth'):file.unlink()
    for filename in ('history.jsonl','train.log','metrics.json'):
        (stage_result/filename).unlink(missing_ok=True)
    log=create_logger(stage_result/'train.log',logger_name=f'C2Seg.StageI.{config.run_name}',console=False,append=False)
    _atomic_json(stage_result/'config.json',{'selected_epoch':epoch_selected, 'stage1':requested_settings})
    tic=time.perf_counter()
    device=device_for(config)
    seed_everything(config.seed,config.deterministic)
    train_loader,val_loader,sampler=_make_stage1_loaders(config)
    clip.to(device);clip.eval()
    with torch.no_grad():
        train_text=_text_bank(clip,train_loader.dataset,device)
        val_text=_text_bank(clip,val_loader.dataset,device)
    clip.cpu()
    if device.type=='cuda':torch.cuda.empty_cache()
    center=torch.nn.functional.normalize(train_text.float(),dim=-1).mean(0,keepdim=True)
    train_teacher=_teacher_for_split(train_text,config,center)
    val_teacher=_teacher_for_split(val_text,config,center)
    neighbour_counts=(train_teacher.quality>0).sum(1)
    semantic_diagnostics={
        'unique_train_texts':int(len(train_text)),
        'unique_val_texts':int(len(val_text)),
        'mean_trusted_neighbours':float(neighbour_counts.float().mean()),
        'fraction_with_trusted_neighbours':float((neighbour_counts>0).float().mean()),
    }
    stage_encoder = encoder
    stage_encoder.to(device)
    heads=Stage1Heads(dim=train_text.shape[-1]).to(device)
    optimizer=torch.optim.AdamW([
        {'params':list(stage_encoder.stage0.parameters())+list(stage_encoder.stage1.parameters())+
                  list(stage_encoder.stage2.parameters()),'lr':config.stage1_lr_cnn,'base_lr':config.stage1_lr_cnn},
        {'params':list(stage_encoder.stage3.parameters())+list(stage_encoder.stage4.parameters()),
         'lr':config.stage1_lr_kan,'base_lr':config.stage1_lr_kan},
        {'params':list(heads.parameters()),'lr':config.stage1_lr_head,'base_lr':config.stage1_lr_head},
    ],weight_decay=config.stage1_weight_decay)
    enabled,dtype=amp_mode(config,device)
    try:scaler=torch.amp.GradScaler('cuda',enabled=enabled and dtype==torch.float16)
    except (AttributeError,TypeError):scaler=torch.cuda.amp.GradScaler(enabled=enabled and dtype==torch.float16)
    history=stage_result/'history.jsonl'
    val_loss_by_epoch={}
    log.info('Stage I | train=%d val=%d batch=256 | schedule=%d train=%d saved=%s',
             len(train_loader.dataset),len(val_loader.dataset),config.stage1_epochs,
             config.stage1_train_epochs,cache_epochs)
    try:
        for epoch in range(1,int(config.stage1_train_epochs)+1):
            factor=stage1_lr_factor(epoch,config)
            for group in optimizer.param_groups:
                group['lr']=max(float(config.stage1_eta_min),float(group['base_lr'])*factor)
            sampler.set_epoch(epoch)
            score_by_phase={}
            for phase,loader,teacher,training in (('train',train_loader,train_teacher,True),
                                                  ('val',val_loader,val_teacher,False)):
                stage_encoder.train(training);heads.train(training)
                count=0;agg={'loss':0.,'global':0.,'region':0.,'separation':0.,'margin':0.}
                for batch in loader:
                    images=batch['image'].to(device,non_blocking=True)
                    masks=batch['mask'].to(device,non_blocking=True)
                    ids=batch['text_id'].to(device,non_blocking=True)
                    if training:
                        if len(images)!=256:raise RuntimeError('Stage I optimizer batch is not 256')
                        optimizer.zero_grad(set_to_none=True)
                    if len(images)<2:continue
                    with torch.set_grad_enabled(training):
                        with torch.autocast(device_type=device.type,enabled=enabled,dtype=dtype):
                            features=stage_encoder(images)
                            with torch.autocast(device_type=device.type,enabled=False):
                                loss,diag=stage1_objective(features,masks,ids,teacher,heads,config)
                        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite Stage I loss epoch {epoch}')
                        if training:
                            if scaler.is_enabled():
                                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(list(stage_encoder.parameters())+list(heads.parameters()),1.)
                                scaler.step(optimizer);scaler.update()
                            else:
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(list(stage_encoder.parameters())+list(heads.parameters()),1.)
                                optimizer.step()
                    n=len(images);count+=n
                    agg['loss']+=float(loss.detach())*n
                    for name in ('global','region','separation','margin'):agg[name]+=diag[name]*n
                if count<1:raise RuntimeError(f'Stage I {phase} had no usable samples')
                score_by_phase[phase]={k:v/count for k,v in agg.items()}
            saved=epoch in cache_epochs
            if saved:
                val_loss_by_epoch[str(epoch)]=score_by_phase['val']['loss']
                _torch_save(cache_weight/f'epoch_{epoch:03d}.pth',{
                    'version':STAGE1_VERSION,'epoch':epoch,
                    'encoder':{k:v.detach().cpu() for k,v in stage_encoder.state_dict().items()}})
            row={'epoch':epoch,'lr':[group['lr'] for group in optimizer.param_groups],
                 'train':score_by_phase['train'],'val':score_by_phase['val'],'saved':saved}
            with history.open('a',encoding='utf-8') as file:
                file.write(json.dumps(row,allow_nan=False)+'\n')
            for line in (f'Stage I Epoch {epoch:03d}/{config.stage1_train_epochs:03d} | Train | Loss={score_by_phase["train"]["loss"]:.5f}',
                         f'Stage I Epoch {epoch:03d}/{config.stage1_train_epochs:03d} | Val   | Loss={score_by_phase["val"]["loss"]:.5f} Margin={score_by_phase["val"]["margin"]:.4f}',
                         f'Stage I Epoch {epoch:03d}/{config.stage1_train_epochs:03d} | Save  | {"encoder checkpoint" if saved else "--"}'):
                log.info('%s',line)
                if logger:logger.info('%s',line)
        meta={'status':'completed',
              'trained_epochs':int(config.stage1_train_epochs),'saved_epochs':cache_epochs,
              'val_loss_by_epoch':val_loss_by_epoch,
              'training_seconds':time.perf_counter()-tic,'last_val_loss':score_by_phase['val']['loss'],
              'global_batch_size':256,
              'steps_per_epoch':len(train_loader),'grad_accumulation':False,
              'train_samples':len(train_loader.dataset),'val_samples':len(val_loader.dataset),
              'semantic_diagnostics':semantic_diagnostics}
        _atomic_json(cache_result/'config.json',requested_settings)
        _atomic_json(cache_meta,meta)
        payload=_load_encoder(selected_cache,encoder,epoch_selected)
        shutil.copy2(selected_cache,local_ckpt)
        _atomic_json(meta_file,{**meta,'selected_epoch':epoch_selected,
                                'checkpoint':str(local_ckpt),'reused':False,
                                'selected_val_loss':val_loss_by_epoch.get(str(epoch_selected))})
        if logger:logger.info('Stage I completed | selected=E%d from %s',epoch_selected,cache_weight)
        return {'checkpoint':str(local_ckpt),
                'selected_epoch':epoch_selected,'reused':False,
                'training_seconds':meta['training_seconds']}
    except Exception:
        log.exception('Stage I failed')
        raise
    finally:
        for handler in list(log.handlers):
            handler.flush();handler.close();log.removeHandler(handler)
        stage_encoder.cpu();heads.cpu()
        if device.type=='cuda':torch.cuda.empty_cache()

import yaml
import json
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from get_model import Model
from baselines import LRIBern, LRIGaussian, Grad, BernMaskP, BernMask, PointMask
from utils import to_cpu, log_epoch, get_data_loaders, get_precision_at_k_and_avgauroc_and_angles, set_seed, init_metric_dict, update_and_save_best_epoch_res, load_checkpoint, ExtractorMLP, get_optimizer


def negative_augmentation(data, data_config, phase, data_loader, batch_idx, loader_len):
    # only used for PLBind dataset
    if data_config['data_name'] == 'plbind':
        neg_aug_p = data_config['neg_aug_p']
        if neg_aug_p and np.random.rand() < neg_aug_p and batch_idx != loader_len - 1 and phase == 'train':
            aug_data = next(iter(data_loader))
            data.x_lig = aug_data.x_lig
            data.pos_lig = aug_data.pos_lig
            data.x_lig_batch = aug_data.x_lig_batch
            data.y = torch.zeros_like(data.y)
    return data


def eval_one_batch(baseline, optimizer, data, epoch, warmup, phase, method_name):
    with torch.set_grad_enabled(method_name in ['gradcam', 'gradgeo', 'bernmask']):
        assert optimizer is None
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # calc angle
        # do_sampling = True if phase == 'valid' and method_name == 'lri_gaussian' else False

        # BernMaskP
        do_sampling = True if phase == 'valid' and method_name == 'bernmask_p' else False # we find this is better for BernMaskP
        loss, loss_dict, org_clf_logits, masked_clf_logits, node_attn, covar_mat, node_noise = baseline.forward_pass(data, epoch, warmup=warmup, do_sampling=do_sampling)
        return loss_dict, to_cpu(org_clf_logits), to_cpu(masked_clf_logits), to_cpu(node_attn), to_cpu(covar_mat), to_cpu(node_noise)


def train_one_batch(baseline, optimizer, data, epoch, warmup, phase, method_name):
    baseline.extractor.train() if hasattr(baseline, 'extractor') else None
    baseline.clf.train() if (method_name != 'bernmask_p' or warmup) else baseline.clf.eval()

    loss, loss_dict, org_clf_logits, masked_clf_logits, node_attn, covar_mat, node_noise = baseline.forward_pass(data, epoch, warmup=warmup, do_sampling=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_dict, to_cpu(org_clf_logits), to_cpu(masked_clf_logits), to_cpu(node_attn), to_cpu(covar_mat), to_cpu(node_noise)


def run_one_epoch(baseline, optimizer, data_loader, epoch, phase, warmup, seed, signal_class, topk, writer, data_config, method_name):
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar
    log_dict = {k: [] for k in ['exp_labels', 'attn', 'clf_labels', 'org_clf_logits', 'masked_clf_logits', 'prec_at_k', 'prec_at_2k', 'prec_at_3k', 'avg_auroc', 'angles', 'eigen_ratio']}
    all_loss_dict = {}

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)

        loss_dict, org_clf_logits, masked_clf_logits, attn, covar_mat, _ = run_one_batch(baseline, optimizer, data.to(baseline.device), epoch, warmup, phase, method_name)
        node_dir, poi_idx = to_cpu(data.get('node_dir', None)), to_cpu(data.get('poi_idx', None))
        exp_labels, clf_labels, prec_at_k, prec_at_2k, prec_at_3k, avg_auroc, angles, eigen_ratio = to_cpu(data.node_label), to_cpu(data.y), [], [], [], [], [], []

        if not warmup:
            exp_labels, attn, covar_mat, node_dir, attn_graph_id = get_relevant_nodes(exp_labels, attn, covar_mat, node_dir, data.batch.reshape(-1), data.y, signal_class)
            prec_at_k, prec_at_2k, prec_at_3k, avg_auroc, angles,  _, eigen_ratio, _ = get_precision_at_k_and_avgauroc_and_angles(exp_labels, attn, covar_mat, node_dir, topk, attn_graph_id)

        for key in log_dict.keys():
            if eval(key) is not None or warmup:
                log_dict[key].append(eval(key))
            else:
                assert key in ['angles', 'eigen_ratio']
                log_dict[key].append(torch.tensor([-1.0]))

        desc = log_epoch(epoch, phase, loss_dict, log_dict, seed, writer, warmup, batch=True)[0]
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v

        if idx == loader_len - 1:
            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc, prec_at_k, prec_at_2k, prec_at_3k, angles, eigen_ratio, avg_loss = log_epoch(epoch, phase, all_loss_dict, log_dict, seed, writer, warmup, batch=False)
        pbar.set_description(desc)
    return org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc, prec_at_k, prec_at_2k, prec_at_3k, angles, eigen_ratio, avg_loss


def get_relevant_nodes(exp_labels, attn, covar_mat, node_dir, attn_graph_id, y, signal_class):
    if signal_class is not None:
        in_signal_class = (y[attn_graph_id] == signal_class).reshape(-1)
        exp_labels, attn, attn_graph_id = exp_labels[in_signal_class], attn[in_signal_class], attn_graph_id[in_signal_class]
        if node_dir is not None:
            node_dir = node_dir[in_signal_class]
        if covar_mat is not None:
            covar_mat = covar_mat[in_signal_class]
    return exp_labels, attn, covar_mat, node_dir, attn_graph_id


def train(config, method_name, model_name, seed, dataset_name, log_dir, device, args):
    writer = SummaryWriter(log_dir) if log_dir is not None else None
    topk = config['logging']['topk']
    if args.sample_tracks>0:
        config['data']['sample_tracks']=args.sample_tracks

    batch_size = config['optimizer']['batch_size']
    epochs = config[method_name]['epochs']
    warmup = config[method_name]['warmup']
    print("########### ",warmup)
    data_config = config['data']
    
    
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, seed)
    if "track" in dataset_name:
        tracks = {}
        for data in dataset:
            for tid in data['track_ids']:
                tracks[tid.item()] = tracks.get(tid.item(),0)+1
        print(tracks)
        tracks = {}
        for data in test_set:
            for tid in data['track_ids']:
                tracks[tid.item()] = tracks.get(tid.item(),0)+1
        print(tracks)
    signal_class = dataset.signal_class
    
    data_dir = Path(data_config['data_dir'])
    model_save_dir = data_dir / dataset_name.split("_")[0]
    mt=args.multi_linear if args.sel_multi_linear<0 else args.sel_multi_linear
    ie=config[method_name].get('info_loss_coef',-1)
    r=config[method_name].get('final_r',-1)
    dr=config[method_name].get('decay_r',-1)
    di=config[method_name].get('decay_interval',-1)
    st=config[method_name].get('sampling_trials',-1)
    config['model_name']=model_name
    config['method_name']=method_name

    sel_dataset = args.sel_dataset if len(args.sel_dataset)>0 else dataset_name
    # sample_tracks = args.sample_tracks if args.sample_tracks>0 else config[method_name]['sample_tracks']
    if args.sample_tracks>0 and config[method_name]['save_mcmc']:
        sel_dataset+=str(args.sample_tracks)
    if args.sel_sample_tracks>0 and config[method_name]['from_mcmc']:
        sel_dataset+=str(args.sel_sample_tracks)
    pre_model_name=f"{sel_dataset}_mt{mt}_{model_name}_ie{ie}_r{r}dr{dr}di{di}st{st}"
    if args.warmup>0:
        pre_model_name += f"wp{args.warmup}"
    if args.init_r>0:
        pre_model_name += f"ir{args.init_r}"
    pre_model_name += f"sd{seed}"
    config[method_name]['mcmc_dir'] = model_save_dir
    config[method_name]['pre_model_name'] = pre_model_name
    if config[method_name]['from_mcmc']:
        clf = Model(model_name, config['model'][model_name], method_name, config[method_name], dataset)#.to(device)
        extractor = ExtractorMLP(config['model'][model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) if 'grad' not in method_name else nn.Identity()
        pred_model_name_clf = f"{model_save_dir}/{pre_model_name}_clf_mcmc.pt"
        print(f'[INFO] Attemping to load a pre-trained MCMC model from {pred_model_name_clf}')
        checkpoint = torch.load(pred_model_name_clf,map_location=device)
        clf.load_state_dict(checkpoint['model_state_dict'])
        pred_model_name_att = f"{model_save_dir}/{pre_model_name}_att_mcmc.pt"
        print(f'[INFO] Attemping to load a pre-trained MCMC model from {pred_model_name_att}')
        checkpoint = torch.load(pred_model_name_att,map_location=device)
        extractor.load_state_dict(checkpoint['model_state_dict'])
        extractor = extractor.to(device)
        clf = clf.to(device)
    else:
        clf = Model(model_name, config['model'][model_name], method_name, config[method_name], dataset).to(device)
        extractor = ExtractorMLP(config['model'][model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) if 'grad' not in method_name else nn.Identity()
        extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits
    if method_name == 'lri_bern':
        baseline = LRIBern(clf, extractor, criterion, config['lri_bern'])
    elif method_name == 'lri_gaussian':
        baseline = LRIGaussian(clf, extractor, criterion, config['lri_gaussian'])
    elif method_name == 'gradgeo':
        baseline = Grad(clf, signal_class, criterion, config['gradgeo'])
    elif method_name == 'gradcam':
        baseline = Grad(clf, signal_class, criterion, config['gradcam'])
    elif method_name == 'bernmask':
        baseline = BernMask(clf, extractor, criterion, config['bernmask'])
    elif method_name == 'bernmask_p':
        baseline = BernMaskP(clf, extractor, criterion, config['bernmask_p'])
    elif method_name == 'pointmask':
        baseline = PointMask(clf, extractor, criterion, config['pointmask'])
    else:
        raise ValueError('Unknown method: {}'.format(method_name))
    if not args.from_mcmc:
        optimizer = get_optimizer(clf, extractor, config['optimizer'], config[method_name], warmup=True)
        metric_dict = deepcopy(init_metric_dict)
        for epoch in range(warmup):
            train_res = run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'train', warmup, seed, signal_class, topk, writer, data_config, method_name)
            valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'valid', warmup, seed, signal_class, topk, writer, data_config, method_name)
            test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, signal_class, topk,  writer, data_config, method_name)
            metric_dict = update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, log_dir, seed, topk, True, writer, args)
    else:
        print("Skipping warmup training")
    if method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
        load_checkpoint(baseline, log_dir, model_name='wp_model')
        if 'grad' in method_name: baseline.start_tracking()

    warmup = 0
    metric_dict = deepcopy(init_metric_dict)
    if not args.from_mcmc:
        clf.emb_model = deepcopy(clf.model) if not config[method_name].get('one_encoder', True) else None
        new_clf=None
        optimizer = get_optimizer(clf, extractor, config['optimizer'], config[method_name], warmup=False)
    else:
        # clf = Model(model_name, config['model'][model_name], method_name, config[method_name], dataset).to(device)
        new_clf = Model(model_name, config['model'][model_name], method_name, config[method_name], dataset).to(device)
        baseline.new_clf = new_clf
        if baseline.multilinear in [5558, 5778, 5550, 5553]:
            baseline.new_clf.load_state_dict(baseline.clf.state_dict())
        optimizer = get_optimizer(new_clf, extractor, config['optimizer'], config[method_name], warmup=True)
        
    if args.test_only:
        epoch = epochs
        train_res = run_one_epoch(baseline, None, loaders['train'], epoch, 'test', warmup, seed, signal_class, topk, writer, data_config, method_name)
        valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'test', warmup, seed, signal_class, topk, writer, data_config, method_name)
        test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, signal_class, topk,  writer, data_config, method_name)
        metric_dict = update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, log_dir, seed, topk, False, writer,args,config[method_name],clf,extractor)
        report_dict = {k.replace('metric/best_', ''): v for k, v in metric_dict.items()}  # for better readability
    else:
        import time
        for epoch in range(epochs):
            if method_name in ['gradcam', 'gradgeo', 'bernmask']:
                if method_name == 'bernmask':
                    train_res = None
                else:
                    train_res = run_one_epoch(baseline, None, loaders['train'], epoch, 'test', warmup, seed, signal_class, topk, writer, data_config, method_name)
                valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'test', warmup, seed, signal_class, topk, writer, data_config, method_name)
                test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, signal_class, topk,  writer, data_config, method_name)
                if train_res is None:
                    train_res = valid_res
            else:
                start_time = time.time()
                train_res = run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'train', warmup, seed, signal_class, topk, writer, data_config, method_name)
                end_time = time.time()
                writer.add_scalar('gsat_time/train', end_time-start_time, epoch)
                start_time = time.time()
                valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'valid', warmup, seed, signal_class, topk, writer, data_config, method_name)
                end_time = time.time()
                writer.add_scalar('gsat_time/val', end_time-start_time, epoch)
                start_time = time.time()
                test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, signal_class, topk,  writer, data_config, method_name)
                end_time = time.time()
                writer.add_scalar('gsat_time/test', end_time-start_time, epoch)
            metric_dict = update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, log_dir, seed, topk, False, writer,args,config[method_name],clf,extractor)
            report_dict = {k.replace('metric/best_', ''): v for k, v in metric_dict.items()}  # for better readability
    return report_dict


def run_one_seed(dataset_name, method_name, model_name, cuda_id, seed, note, time,args):
    set_seed(seed)
    config_name = dataset_name.split('_')[0]
    sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') /  f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    config[f'{method_name}']['multi_linear'] = args.multi_linear
    config[f'{method_name}']['sampling_trials'] = args.sampling_trials
    
    config[f'{method_name}']['from_mcmc'] = args.from_mcmc
    config[f'{method_name}']['sel_r'] = args.sel_r
    config[f'{method_name}']['save_mcmc'] = args.save_mcmc
    config['logging']['tensorboard'] = args.tensorboard
    config['logging']['best_acc'] = args.best_acc
    
    
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])
    if args.decay_interval>=0:
        config[method_name]['decay_interval'] = args.decay_interval
    if args.final_r>0:
        config[method_name]['final_r'] = args.final_r
    if args.init_r>0:
        config[method_name]['init_r'] = args.init_r
    
    if args.warmup>0:
        config[method_name]['warmup']= args.warmup
    if args.info_loss_coef>=0:
        config[f'{method_name}']['info_loss_coef'] = args.info_loss_coef
        # print(config[f'{method_name}']['info_loss_coef'])
    print('-' * 80), print('-' * 80)
    print(f'Config: ', json.dumps(config, indent=4))
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    log_dir = None
    if config['logging']['tensorboard'] or method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
        log_dir = Path(config['data']['data_dir']) / config_name / f'logs{sub_dataset_name}' / ('-'.join([time, method_name, model_name, 'seed'+str(seed), note]))
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, log_dir / config_path.name)
    report_dict = train(config, method_name, model_name, seed, dataset_name, log_dir, device, args)
    return report_dict


def main():
    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S.%f")[:-3]
    parser = argparse.ArgumentParser(description='Train SAT')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack_2T')
    parser.add_argument('-m', '--method', type=str, help='method used', default='lri_gaussian')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('-ld', '--log_dir', type=str, help='backbone used', default='logs')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('-sd','--seed', type=int, help='random seed', default=-1)
    parser.add_argument('-di','--decay_interval', type=int, help='random seed', default=-1)
    parser.add_argument('-mt','--multi_linear', type=int, help='random seed', default=0)
    parser.add_argument('-smt','--sel_multi_linear', type=int, help='random seed', default=-1)
    parser.add_argument('-sda','--sel_dataset', type=str, help='random seed', default="")
    parser.add_argument('-tk','--sample_tracks', type=int, help='random seed', default=-1)
    parser.add_argument('-stk','--sel_sample_tracks', type=int, help='random seed', default=-1)
    parser.add_argument('-ie','--info_loss_coef', type=float, help='random seed', default=-1)
    parser.add_argument('-fr','--final_r', type=float, help='random seed', default=-1)
    parser.add_argument('-ir','--init_r', type=float, help='random seed', default=-1)
    parser.add_argument('-sr','--sel_r', type=float, help='random seed', default=-1)
    parser.add_argument('-st','--sampling_trials', type=int, help='random seed', default=0)
    parser.add_argument('-wp','--warmup', type=int, help='random seed', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('-ba','--best_acc',action='store_true')
    parser.add_argument('-fm','--from_mcmc',action='store_true')
    parser.add_argument('-sm','--save_mcmc',action='store_true')
    parser.add_argument('-tb','--tensorboard',action='store_true')
    parser.add_argument('-to','--test_only',action='store_true')
    args = parser.parse_args()

    print(args)
    metric_dicts = []
    if args.seed>=0:
        num_seeds = 1
    else:
        num_seeds = 5
    for seed in range(num_seeds):
        if args.seed>=0:
            seed = args.seed
        report_dict = run_one_seed(args.dataset, args.method, args.backbone, args.cuda, seed, args.note, time,args)
        print(json.dumps(report_dict, indent=4))
        metric_dicts.append(report_dict)
    print(f"Final metrics")
    final_metrics = {}
    metric_keys = metric_dicts[0].keys()
    for key in metric_keys:
        metric_values = np.array([metric[key] for metric in metric_dicts])
        if "tvd" not in key:
            print(f"{key}:{metric_values.mean()*100:2.2f}+-{metric_values.std()*100:2.2f}")
        else:
            print(f"{key}:{metric_values.mean():2.2f}+-{metric_values.std():2.2f}")
    


if __name__ == '__main__':
    main()

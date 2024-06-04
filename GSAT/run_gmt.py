import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected, to_undirected
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import copy
import torch_geometric.data.batch as DataBatch
from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data, relabel
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config,model_config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        self.model_name = model_config['model_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.cur_pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']
        self.cur_info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)
        self.sel_r = method_config.get('sel_r', 0.5)

        self.from_scratch = method_config['from_scratch']
        self.save_mcmc = method_config.get('save_mcmc',False)
        self.from_mcmc = method_config.get('from_mcmc',False)
        self.multi_linear = method_config.get('multi_linear',False)
        self.mcmc_dir = method_config['mcmc_dir']
        self.pre_model_name = method_config['pre_model_name'] 

        if self.multi_linear in [5552]:
            self.fc_proj = nn.Sequential(nn.Sequential(nn.Dropout(p=0.33),
                                nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                                nn.BatchNorm1d(self.clf.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                            ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters())+ list(self.fc_proj.parameters()), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)
        if self.multi_linear in [5553,5554]:
            class_dim = 1 if num_class == 2 and not multi_label else num_class
            self.fc_proj = nn.Sequential(
                                nn.Sequential(nn.Linear(self.clf.hidden_size, class_dim),
                                nn.BatchNorm1d(class_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(class_dim, class_dim),
                            ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters())+ list(self.fc_proj.parameters()), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

        if self.multi_linear in [5550,5552,5553,5554,5555,5559,5449,5229,5669]:
            if not self.from_mcmc:
                self.fc_out = self.clf
            self.fc_out = get_model(model_config['x_dim'], model_config['edge_attr_dim'], num_class, model_config['multi_label'], model_config, device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            if self.multi_linear in [5552,5554,5555,5669,5449]:
                self.fc_out.load_state_dict(copy.deepcopy(self.clf.state_dict()))
            self.optimizer = torch.optim.Adam(self.fc_out.parameters(), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

        self.sampling_trials = method_config.get('sampling_trials',100)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)


    def __loss__(self, att, clf_logits, clf_labels, epoch,training=False,agg='mean'):
        if clf_logits.size(0)!=clf_labels.size(0):
            pred_losses = []
            for i in range(clf_logits.size(0)):
                pred_losses.append(self.criterion(clf_logits[i,:], clf_labels))
            if agg.lower() == 'max':
                pred_loss = torch.stack(pred_losses).max()
            else:
                pred_loss = torch.stack(pred_losses).mean()
        else:
            pred_losses = None
            pred_loss = self.criterion(clf_logits,clf_labels)
        
        r = self.final_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
        pred_lossc = pred_loss * self.pred_loss_coef
        info_lossc = info_loss * self.cur_info_loss_coef
        loss = pred_lossc + info_lossc
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        if pred_losses != None:
            for i,pl in enumerate(pred_losses):
                loss_dict[f'pred_L{i}'] = pl.item()
        if training:
            self.optimizer.zero_grad()
            pred_lossc.backward(retain_graph=True)
            pred_grad = []
            for param in self.extractor.parameters():
                if param.grad != None:
                    pred_grad.append(param.grad.data.clone().flatten().detach())
            pred_grad = torch.cat(pred_grad) if len(pred_grad)>0 else torch.zeros([1]).to(loss.device)
            self.optimizer.zero_grad()
            info_lossc.backward(retain_graph=True)
            info_grad = []
            for param in self.extractor.parameters():
                if param.grad != None:
                    info_grad.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            info_grad = torch.cat(info_grad) if len(pred_grad)>0 else torch.zeros([1])
            grad_sim = F.cosine_similarity(pred_grad.unsqueeze(0),info_grad.unsqueeze(0)).to(loss.device)
            loss_dict['grad_sim']=grad_sim.item()
            loss_dict['pred_grad']=pred_grad.norm().item()
            loss_dict['info_grad']=info_grad.norm().item()
        return loss, loss_dict
    def package_subgraph(self, data, att_bern, epoch,verbose=False):
        b = torch.bernoulli(att_bern)
        att_binary = (b - att_bern).detach() + att_bern  # straight-through estimator
        # return att_binary
        def relabel(x, edge_index, batch, pos=None):
            num_nodes = x.size(0)
            sub_nodes = torch.unique(edge_index)
            x = x[sub_nodes]
            batch = batch[sub_nodes]
            row, col = edge_index
            # remapping the nodes in the explanatory subgraph to new ids.
            node_idx = row.new_full((num_nodes,), -1)
            node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
            edge_index = node_idx[edge_index]
            if pos is not None:
                pos = pos[sub_nodes]
            return x, edge_index, batch, pos

        idx_reserve = torch.nonzero(att_binary == 1, as_tuple=True)[0]
        idx_drop = torch.nonzero(att_binary == 0, as_tuple=True)[0]
        if verbose:
            print(len(idx_reserve)/len(att_binary), self.get_r(
                self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r))
        causal_edge_index = data.edge_index[:, idx_reserve]
        if data.edge_attr is not None:
            causal_edge_attr = data.edge_attr[idx_reserve]
        else:
            causal_edge_attr = None
        causal_edge_atten = att_binary[idx_reserve]
        causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)
        graph_prob = 0
        return causal_x, causal_edge_index, causal_batch, causal_edge_attr, causal_edge_atten, graph_prob
    def attend(self,data,att_log_logits, epoch, training):
        att = self.sampling(att_log_logits, epoch, training)
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                # not for spomtif
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            # molhiv
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
        return edge_att

    def split_graph(self,data, edge_score, ratio):
        # Adopt from GOOD benchmark to improve the efficiency
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score.view(-1), data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = -edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return new_idx_reserve, new_idx_drop

    def forward_pass(self, data, epoch, training):

        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)


        if self.multi_linear==3:
            edge_att = self.attend(data,att_log_logits, epoch, training)
            clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att,att_opt='first')
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training)
            return edge_att, loss, loss_dict, clf_logits
        elif self.multi_linear==5:
            sampling_logits = []
            sampling_trials = self.sampling_trials
            while len(sampling_logits)<sampling_trials:
                edge_att = self.attend(data,att_log_logits, epoch, training)
                # cur_edge_att = self.binarize_att(data,edge_att)
                b = torch.bernoulli(edge_att)
                cur_edge_att = (b - edge_att).detach() + edge_att  # straight-through estimator
                clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=cur_edge_att)
                sampling_logits.append(clf_logits)
            clf_logits = torch.stack(sampling_logits).mean(dim=0)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training)
        elif self.multi_linear==8:
            sampling_logits = []
            cur_r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            sampling_trials = self.sampling_trials
            if cur_r==1 or self.decay_interval>epoch:
                sampling_trials = 1
                self.cur_info_loss_coef = 2*self.info_loss_coef
            else:
                sampling_trials = self.sampling_trials
                self.cur_info_loss_coef = self.info_loss_coef
            while len(sampling_logits)<sampling_trials:
                edge_att = self.attend(data,att_log_logits, epoch, training)
                b = torch.bernoulli(edge_att)
                cur_edge_att = (b - edge_att).detach() + edge_att  # straight-through estimator
                clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=cur_edge_att)
                sampling_logits.append(clf_logits)
            clf_logits = torch.stack(sampling_logits).mean(dim=0)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training)
        elif self.multi_linear==5550:
            att_log_logits = att_log_logits.detach()
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_att,self.sel_r)
            causal_edge_weight = edge_att
            # causal_edge_weight[new_idx_reserve] = 1
            causal_edge_weight[new_idx_drop] = 0
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5552:
            att_log_logits = att_log_logits.detach()
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_att,self.sel_r)
            causal_edge_weight = edge_att
            # causal_edge_weight[new_idx_reserve] = 1
            causal_edge_weight[new_idx_drop] = 0
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5553:
            att_log_logits = att_log_logits.detach()
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_att,self.sel_r)
            causal_edge_weight = edge_att
            causal_edge_weight[new_idx_reserve] = 1
            # causal_edge_weight[new_idx_drop] = 0
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5554:
            att_log_logits = att_log_logits.detach()
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_att,self.sel_r)
            causal_edge_weight = edge_att
            causal_edge_weight[new_idx_reserve] = 1
            # causal_edge_weight[new_idx_drop] = 0
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5555:
            att_log_logits = att_log_logits.detach()
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_att,self.sel_r)
            causal_edge_weight = edge_att
            causal_edge_weight[new_idx_reserve] = 1
            causal_edge_weight[new_idx_drop] = 0
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5559:
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
            # clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5669:
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5449:
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att,att_opt='first')
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        elif self.multi_linear==5229:
            edge_att = self.attend(data,att_log_logits, epoch, training=False)
            clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att,att_opt='first')
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training=False)
            loss = self.criterion(clf_logits,data.y)
        else:
            edge_att = self.attend(data,att_log_logits, epoch, training)
            clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
            loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training)
        edge_att = att_log_logits.sigmoid().detach()
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, edge_att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (edge_att + trans_val_perm) / 2
            else:
                edge_att = edge_att
        else:
            edge_att = self.lift_node_att_to_edge_att(edge_att, data.edge_index)

        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        self.eval()
        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        self.train()
        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                                                                                        all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)
        return att_auroc, precision, clf_acc, clf_roc, avg_loss, all_loss_dict.get('logits_tvd',-1)

    def train_self(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            if self.multi_linear==5775:
                valid_res = self.run_one_epoch(loaders['valid'], epoch, 'test', use_edge_attr)
                print(f"validation: att_auroc {valid_res[0]}, precision {valid_res[1]}, clf_acc {valid_res[2]}, clf_roc {valid_res[3]}")
                train_res = test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
                print(f"test: att_auroc {test_res[0]}, precision {test_res[1]}, clf_acc {test_res[2]}, clf_roc {test_res[3]}")
            else:
                train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
                valid_res = self.run_one_epoch(loaders['valid'], epoch, 'val', use_edge_attr)
                test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('xgnn_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 6
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if (r == self.final_r or self.fix_r)  and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1],
                               'metric/best_tvd_train':train_res[5],'metric/best_tvd_valid':valid_res[5],'metric/best_tvd_test':test_res[5],}
                if self.save_mcmc:
                    # mcmc_dir = Path("/".join(str(self.model_dir).split("/")[:3]))
                    save_checkpoint(self.clf, self.mcmc_dir, model_name=self.pre_model_name+f"_clf_mcmc")
                    save_checkpoint(self.extractor, self.mcmc_dir, model_name=self.pre_model_name+f"_att_mcmc")
                # save_checkpoint(self.clf, self.model_dir, model_name='xgnn_clf_epoch_' + str(epoch))
                # save_checkpoint(self.extractor, self.model_dir, model_name='xgnn_att_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'xgnn_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    try:
                        self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)
                    except Exception as e:
                        print(e)

            # if epoch == self.epochs - 1:
            #     save_checkpoint(self.clf, self.model_dir, model_name='xgnn_clf_epoch_' + str(epoch))
            #     save_checkpoint(self.extractor, self.model_dir, model_name='xgnn_att_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}, Best Test Logits TVD: {metric_dict["metric/best_tvd_test"]:.3f}')
            print('====================================')
            print('====================================')
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: xgnn_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: xgnn_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'xgnn_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        if 'logits_tvd' in loss_dict.keys() and not batch:
            self.writer.add_scalar(f'xgnn_{phase}/tvd_gap_acc', att_auroc-clf_acc, epoch)
            self.writer.add_scalar(f'xgnn_{phase}/tvd_gap_roc', att_auroc-clf_roc, epoch)
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.writer.add_histogram(f'xgnn_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'xgnn_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'xgnn_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_scalar(f'xgnn_{phase}/avg_att_weights_std/', att.std(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/xgnn_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att
    @staticmethod
    def lift_edge_att_to_node_att(edge_att, edge_index,size=None):
        src_att = edge_att[edge_index[0]].view(-1)
        dst_att = edge_att[edge_index[1]].view(-1)
        # print(src_att.size(),edge_index[0].size())
        src_att = scatter(src_att,edge_index[0],reduce='mul',dim_size=size)
        src_att = scatter(dst_att,edge_index[0],reduce='mul',dim_size=size)
        node_att =1.0- src_att * src_att
        return node_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_xgnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state,args):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model_config['x_dim'] = x_dim
    model_config['edge_attr_dim'] = edge_attr_dim
    model_config['multi_label'] = aux_info['multi_label']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        pre_model_name = f"{data_dir}/{dataset_name}/"+model_name
        if args.num_layers>0:
            pre_model_name += f"{args.num_layers}L"
        pre_model_name += f'{random_state}.pt'
        try:
            print(f'[INFO] Attemping to load a pre-trained model from {pre_model_name}')
            # load_checkpoint(model, model_dir=f"{data_dir}/{dataset_name}/", model_name=f'seed{random_state}_epoch_{pretrain_epochs}')
            checkpoint = torch.load(pre_model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.force_train:
                raise Exception("[INFO] Forced re-pretraining the model...")
        except Exception as e:
            print('[INFO] Failing to find a pre-trained model. Now pretraining the model...')
            train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                            model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
            # save_checkpoint(model, model_dir=f"{data_dir}/{dataset_name}/", model_name=f'seed{random_state}_epoch_{pretrain_epochs}')
            torch.save({'model_state_dict': model.state_dict()}, pre_model_name)
            # load_checkpoint(model, model_dir=f"{data_dir}/{dataset_name}/", model_name=f'seed{random_state}_epoch_{pretrain_epochs}')
            # checkpoint = torch.load(pre_model_name)
            # model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('[INFO] Training both the model and the attention from scratch...')
    
    #local_config[f'{method_name}_config']['multi_linear'] 
    mt=5
    if args.gcat_multi_linear >= 0:
        mt = args.gcat_multi_linear
    ie=local_config[f'{method_name}_config']['info_loss_coef']
    r=local_config[f'{method_name}_config']['final_r']
    dr=local_config[f'{method_name}_config']['decay_r']

    di=local_config[f'{method_name}_config']['decay_interval']
    st=local_config[f'{method_name}_config']['sampling_trials']
    model_save_dir = data_dir / dataset_name / f'{args.log_dir}'
    pre_model_name=f"{dataset_name}_mt{mt}_{model_name}_scracth{method_config['from_scratch']}_ie{ie}_r{r}dr{dr}di{di}st{st}"
    if args.epochs>0:
        pre_model_name += f"ep{args.epochs}"
    pre_model_name+= f"sd{random_state}"
    
    if method_config['from_mcmc']:
        pred_model_name_clf = f"{model_save_dir}/{pre_model_name}_clf_mcmc.pt"
        print(f'[INFO] Attemping to load a pre-trained MCMC model from {pred_model_name_clf}')
        checkpoint = torch.load(pred_model_name_clf,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
        pred_model_name_att = f"{model_save_dir}/{pre_model_name}_att_mcmc.pt"
        print(f'[INFO] Attemping to load a pre-trained MCMC model from {pred_model_name_att}')
        checkpoint = torch.load(pred_model_name_att,map_location=device)
        extractor.load_state_dict(checkpoint['model_state_dict'])
        from torch_geometric.data import Batch
        from torch_geometric.utils import degree
        print('[INFO] Calculating degree based on extractor...')
        batched_train_set = Batch.from_data_list(loaders['train'].dataset)
        d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=10)
        model_config['deg2'] = aux_info['deg2'] = deg
    if not method_config['from_mcmc']:
        extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config, **method_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    method_config_new = copy.deepcopy(method_config)
    method_config_new['mcmc_dir'] = model_save_dir
    method_config_new['pre_model_name'] = pre_model_name
    xgnn = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config_new, shared_config,model_config)
    metric_dict = xgnn.train_self(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    parser.add_argument('-ld','--log_dir',default='logs', type=str, help='')
    parser.add_argument('-mt','--multi_linear',default=-1, type=int, help='which gmt variant to use, 3 for lin, 5 for sam')
    parser.add_argument('-gmt','--gcat_multi_linear',default=-1, type=int, help='will use it to name the model')
    parser.add_argument('-st','--sampling_trials',default=100, type=int, help='number of sampling rounds')
    parser.add_argument('-fs','--from_scratch',default=-1, type=int, help='from scratch or not')
    parser.add_argument('-fm','--from_mcmc',action='store_true')
    parser.add_argument('-sm','--save_mcmc',action='store_true')
    parser.add_argument('-sd','--seed',default=-1, type=int)
    parser.add_argument('-ie','--info_loss_coef',default=-1, type=float)
    parser.add_argument('-r','--ratio',default=-1, type=float)
    parser.add_argument('-ir','--init_r',default=-1, type=float)
    parser.add_argument('-sr','--sel_r',default=-1, type=float, help='ratio for subgraph decoding')
    parser.add_argument('-dr','--decay_r',default=-1, type=float)
    parser.add_argument('-di','--decay_interval',default=-1, type=int)
    parser.add_argument('-L','--num_layers',default=-1, type=int)
    parser.add_argument('-ep','--epochs',default=-1, type=int)
    parser.add_argument('-ft','--force_train',action='store_true')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))
    if args.epochs>=0:
        local_config[f'{method_name}_config']['epochs'] = args.epochs
    if args.multi_linear>=0:
        local_config[f'{method_name}_config']['multi_linear'] = args.multi_linear

    local_config[f'{method_name}_config']['sampling_trials'] = args.sampling_trials
    if args.from_scratch>=0:
        local_config[f'{method_name}_config']['from_scratch'] = bool(args.from_scratch)
    local_config[f'{method_name}_config']['from_mcmc'] = bool(args.from_mcmc)
    local_config[f'{method_name}_config']['save_mcmc'] = bool(args.save_mcmc)
        
    if args.info_loss_coef>=0:
        local_config[f'{method_name}_config']['info_loss_coef'] = args.info_loss_coef
    if args.ratio>=0:
        local_config[f'{method_name}_config']['final_r'] = args.ratio
    if args.init_r>=0:
        local_config[f'{method_name}_config']['init_r'] = args.init_r
    if args.init_r>=0:
        local_config[f'{method_name}_config']['sel_r'] = args.init_r
    if args.decay_r>=0:
        local_config[f'{method_name}_config']['decay_r'] = args.decay_r
    if args.decay_interval>=0:
        local_config[f'{method_name}_config']['decay_interval'] = args.decay_interval
    if args.num_layers>=0:
        local_config[f'model_config']['num_layers'] = args.num_layers
    print(local_config[f'{method_name}_config'])
    print(local_config[f'model_config'])
    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    if args.seed>=0:
        num_seeds = 1
    for random_state in range(num_seeds):
        if args.seed>=0:
            random_state = args.seed
        log_dir = data_dir / dataset_name / f'{args.log_dir}' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name+\
            f"-fs{args.from_scratch}-mt{args.multi_linear}st{args.sampling_trials}-ie{args.info_loss_coef}-r{local_config[f'{method_name}_config']['final_r']}-dr{local_config[f'{method_name}_config']['decay_r']}-di{local_config[f'{method_name}_config']['decay_interval']}")
        hparam_dict, metric_dict = train_xgnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state,args)
        metric_dicts.append(metric_dict)
    print(f"Final metrics")
    final_metrics = {}
    metric_keys = metric_dicts[0].keys()
    for key in metric_keys:
        metric_values = np.array([metric[key] for metric in metric_dicts])
        if "tvd" not in key:
            print(f"{key}:{metric_values.mean()*100:2.2f}+-{metric_values.std()*100:2.2f}")
        else:
            print(f"{key}:{metric_values.mean():2.2f}+-{metric_values.std():2.2f}")
    # log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    # log_dir.mkdir(parents=True, exist_ok=True)
    # writer = Writer(log_dir=log_dir)
    # write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()

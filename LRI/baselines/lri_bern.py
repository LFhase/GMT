import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np


class LRIBern(nn.Module):

    def __init__(self, clf, extractor, criterion, config, new_clf=None):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.info_loss_coef = config['info_loss_coef']
        self.cur_info_loss_coef = config['info_loss_coef']
        self.temperature = config['temperature']

        self.final_r = config['final_r']
        self.decay_interval = config['decay_interval']
        self.decay_r = config['decay_r']
        self.init_r = config['init_r']
        self.sel_r = config.get('sel_r',0)
        self.multilinear = config.get('multi_linear', 0)
        self.sampling_trials = config.get('sampling_trials',100)
        self.attn_constraint = config['attn_constraint']
        self.from_mcmc = config.get('from_mcmc',False)
        self.save_mcmc = config.get('save_mcmc',True)
        self.new_clf=new_clf
        
    def __loss__(self, attn, clf_logits, clf_labels, epoch, warmup):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        r = self.get_r(epoch)
        info_loss = (attn * torch.log(attn/r + 1e-6) + (1 - attn) * torch.log((1 - attn)/(1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = self.pred_loss_coef * pred_loss
        info_loss = self.cur_info_loss_coef * info_loss

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'r': r}
        return loss, loss_dict

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if self.multilinear in [5558,5778,5669,5550,5553,5552,5554]:
            # self.clf.eval()
            # self.extractor.eval()
            if self.clf.training:
                self.new_clf.train()
            else:
                self.new_clf.eval()
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        (emb, pool_out_lig), edge_index = self.clf.get_emb(data)
        node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=pool_out_lig)

        if self.attn_constraint == 'smooth_min':
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
        else:
            assert self.attn_constraint == 'none'

        node_attn = self.sampling(node_attn_log_logits, do_sampling)
        edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        if self.multilinear==55:
            sampling_logits = []
            sampling_trials = self.sampling_trials
            for i in range(sampling_trials):
                cur_node_attn = self.sampling(node_attn_log_logits, do_sampling)
                cur_edge_attn = self.node_attn_to_edge_attn(cur_node_attn, edge_index)
                b = torch.bernoulli(cur_edge_attn)
                cur_edge_attn = (b - cur_edge_attn).detach() + cur_edge_attn  # straight-through estimator
                masked_clf_logits = self.clf(data, edge_attn=cur_edge_attn, node_attn=cur_node_attn)
                sampling_logits.append(masked_clf_logits)
            masked_clf_logits = torch.stack(sampling_logits).mean(dim=0)
        elif self.multilinear==88:
            sampling_logits = []
            sampling_trials = self.sampling_trials
            if self.get_r(epoch)==1 or epoch<self.decay_interval:
                sampling_trials = 1
                self.cur_info_loss_coef = 2*self.info_loss_coef
            else:
                self.cur_info_loss_coef = self.info_loss_coef
            for i in range(sampling_trials):
                cur_node_attn = self.sampling(node_attn_log_logits, do_sampling)
                cur_edge_attn = self.node_attn_to_edge_attn(cur_node_attn, edge_index)
                b = torch.bernoulli(cur_edge_attn)
                cur_edge_attn = (b - cur_edge_attn).detach() + cur_edge_attn  # straight-through estimator
                masked_clf_logits = self.clf(data, edge_attn=cur_edge_attn, node_attn=cur_node_attn)
                sampling_logits.append(masked_clf_logits)
            masked_clf_logits = torch.stack(sampling_logits).mean(dim=0)
        elif self.multilinear==552:
            sampling_logits = []
            sampling_trials = self.sampling_trials
            for i in range(sampling_trials):
                cur_node_attn = self.sampling(node_attn_log_logits, do_sampling)
                cur_edge_attn = self.node_attn_to_edge_attn(cur_node_attn, edge_index)
                b = torch.bernoulli(cur_edge_attn)
                cur_edge_attn = (b - cur_edge_attn).detach() + cur_edge_attn  # straight-through estimator
                masked_clf_logits = self.clf(data, edge_attn=cur_edge_attn, node_attn=cur_node_attn,att_opt="first")
                sampling_logits.append(masked_clf_logits)
            masked_clf_logits = torch.stack(sampling_logits).mean(dim=0)
        elif self.multilinear in [5550,5553]:
            node_attn = node_attn_log_logits.sigmoid().detach()
            edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
            new_idx_reserve, new_idx_drop = self.split_graph(data,edge_index,edge_attn,self.sel_r)
            causal_edge_weight = edge_attn
            if self.multilinear==5553:
                causal_edge_weight[new_idx_reserve] = 1
            if self.multilinear==5550:
                causal_edge_weight[new_idx_drop] = 0
            
            masked_clf_logits = self.new_clf(data, edge_attn=causal_edge_weight, node_attn=node_attn)
            original_clf_logits = self.clf(data)

            loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch, warmup)
            loss = self.criterion(masked_clf_logits, data.y.float())
            return loss, loss_dict, original_clf_logits, masked_clf_logits, node_attn.reshape(-1), None, None
        elif self.multilinear==2:
            masked_clf_logits = self.clf(data, edge_attn=edge_attn, node_attn=node_attn,att_opt="first")
        else:
            masked_clf_logits = self.clf(data, edge_attn=edge_attn, node_attn=node_attn)
        original_clf_logits = self.clf(data)

        loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch, warmup)
        return loss, loss_dict, original_clf_logits, masked_clf_logits, node_attn.reshape(-1), None, None
    def split_graph(self,data, edge_index,edge_score, ratio):
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
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score.view(-1), data.batch[edge_index[0]], ratio, descending=True)
        return new_idx_reserve, new_idx_drop
    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def sampling(self, attn_log_logits, do_sampling):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            attn_bern = ((attn_log_logits + random_noise) / self.temperature).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn

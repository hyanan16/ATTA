import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from torch import nn
import torch
# import models for resnet18
import itertools
import os
import GTTA.data.loaders.misc as misc
from GTTA import register
from GTTA.utils.config_reader import Conf
from GTTA.utils.initial import reset_random_seed
from GTTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm


@register.alg_register
class TALMSTCluster:
    def __init__(self, config: Conf):
        super(TALMSTCluster, self).__init__()
        reset_random_seed(config)
        self.dataset = register.datasets[config.dataset.name](config.dataset.dataset_root, config.dataset.test_envs,
                                                              config)
        config.dataset.dataset_type = 'image'
        config.dataset.input_shape = self.dataset.input_shape
        config.dataset.num_classes = 1 if self.dataset.num_classes == 2 else self.dataset.num_classes
        config.model.model_level = 'image'
        config.metric.set_score_func(self.dataset.metric)
        config.metric.set_loss_func(self.dataset.task)

        self.config = config

        self.inf_loader = [InfiniteDataLoader(env, weights=None, batch_size=self.config.train.train_bs,
                                              num_workers=self.config.num_workers) for env in self.dataset]
        self.fast_loader = [FastDataLoader(env, weights=None,
                                           batch_size=self.config.train.train_bs,
                                           num_workers=self.config.num_workers) for env in self.dataset]
        self.buffer_fast_loader = [FastDataLoader(env, weights=None,
                                                  batch_size=100,
                                                  num_workers=self.config.num_workers) for env in self.dataset]

        self.encoder = register.models[config.model.name](config).to(self.config.device)
        self.fc = nn.Linear(self.encoder.n_outputs, config.dataset.num_classes).to(self.config.device)
        self.model = nn.Sequential(self.encoder, self.fc).to(self.config.device)
        self.budgets = self.config.atta.budgets
        self.anchors = []
        self.buffer = []
        self.n_clusters = 10
        self.cold_start = self.config.atta.ActiveTTA.cold_start


        if not os.path.exists(self.config.ckpt_dir):
            os.makedirs(self.config.ckpt_dir)

    def __call__(self, *args, **kwargs):
        self.train_on_env(0, train_only_fc=False, train_or_load='load')
        # self.test_on_env(0)
        # self.test_on_env(1)
        # self.test_on_env(2)
        # self.test_on_env(3)
        self.adapt_on_env(1)
        print(self.budgets)
        # self.test_on_env(0)
        # self.test_on_env(1)
        # self.test_on_env(2)
        # self.test_on_env(3)
        self.adapt_on_env(2)
        print(self.budgets)
        # self.test_on_env(0)
        # self.test_on_env(1)
        # self.test_on_env(2)
        # self.test_on_env(3)
        self.adapt_on_env(3)
        print(self.budgets)
        self.cluster_train()
        self.test_on_env(0)
        self.test_on_env(1)
        self.test_on_env(2)
        self.test_on_env(3)


    @torch.no_grad()
    def test_on_env(self, env_id):
        self.encoder.eval()
        self.fc.eval()
        test_loss = 0
        test_acc = 0
        for data, target in self.fast_loader[env_id]:
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.fc(self.encoder(data))
            test_loss += self.config.metric.loss_func(output, target, reduction='sum').item()
            test_acc += self.config.metric.score_func(target, output) * len(data)
        test_loss /= len(self.fast_loader[env_id].dataset)
        test_acc /= len(self.fast_loader[env_id].dataset)
        print(f'Env {env_id} Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')

    @torch.enable_grad()
    def train_on_env(self, env_id, train_only_fc=True, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + f'/encoder_{env_id}.pth'):
            reset_random_seed(config)
            if train_only_fc:
                self.encoder.train()
                self.fc.train()
                optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.train.lr)
                inf_loader = self.extract_pretrained_feat(self.fast_loader[env_id], self.config.train.train_bs)
                for batch_idx, (data, target) in enumerate(inf_loader):
                    optimizer.zero_grad()
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    output = self.fc(data)
                    loss = self.config.metric.loss_func(output, target)
                    acc = self.config.metric.score_func(target, output)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % self.config.train.log_interval == 0:
                        print(f'Iteration: {batch_idx} Loss: {loss.item():.4f} Acc: {acc:.4f}')
                    if batch_idx > self.config.train.max_iters:
                        break
            else:
                self.encoder.train()
                self.fc.train()
                optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.fc.parameters()),
                                             lr=self.config.train.lr)
                for batch_idx, (data, target) in enumerate(self.inf_loader[env_id]):
                    optimizer.zero_grad()
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    output = self.fc(self.encoder(data))
                    loss = self.config.metric.loss_func(output, target)
                    acc = self.config.metric.score_func(target, output)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % self.config.train.log_interval == 0:
                        print(f'Iteration: {batch_idx} Loss: {loss.item():.4f} Acc: {acc:.4f}')
                    if batch_idx > self.config.train.max_iters:
                        break
            torch.save(self.encoder.state_dict(), self.config.ckpt_dir + f'/encoder_{env_id}.pth')
            torch.save(self.fc.state_dict(), self.config.ckpt_dir + f'/fc_{env_id}.pth')
        else:
            self.encoder.load_state_dict(
                torch.load(self.config.ckpt_dir + f'/encoder_{env_id}.pth', map_location=self.config.device))
            self.fc.load_state_dict(torch.load(self.config.ckpt_dir + f'/fc_{env_id}.pth', map_location=self.config.device))


    @torch.enable_grad()
    def naive_test_time_train(self, anchors):
        print('Test time train')
        data, target, feat = zip(*anchors)
        data, target, feat = torch.cat(data), torch.cat(target), torch.cat(feat)
        anchor_loader = InfiniteDataLoader(TensorDataset(data.cpu(), target.cpu()), weights=None,
                                           batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.train.lr)
        for i, (data, target) in enumerate(anchor_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            optimizer.zero_grad()
            output = self.fc(self.encoder(data))
            soft_target = torch.eye(output.shape[1])[target].to(self.config.device)
            alpha = 0.2
            loss = self.config.metric.loss_func(output, target)
            loss.backward()
            optimizer.step()
            if i > 5000:
                break
        data, target, feat = zip(*anchors)
        data, target, feat = torch.cat(data), torch.cat(target), torch.cat(feat)
        outputs = self.fc(self.encoder(data))
        entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        print(entropy.mean(), entropy.std(), entropy.max())


    @staticmethod
    def feature_distance(feat1, feat2, dis_type='l2'):
        if dis_type == 'l2':
            return (feat1 - feat2).pow(2).sum(1).sqrt()
        elif dis_type == 'cos':
            return 1 - (feat1 / feat1.norm()).T @ (feat2 / feat2.norm())
        elif dis_type == 'l1':
            return (feat1 - feat2).abs().sum(1)
        else:
            raise NotImplementedError

    def diverse_dis(self, feat_k, feat_q, dis_type='cos'):
        feat_k = feat_k / feat_k.norm(dim=1, keepdim=True)
        feat_q = feat_q / feat_q.norm(dim=1, keepdim=True)
        dis = 1 - feat_k @ feat_q.T
        return dis

    # def class_anchor_sel(self, class_anchors, feat, ):


    @torch.enable_grad()
    def test_time_train(self, anchors):
        print('Test time train')
        data, feat, pseudo_label, entropy, target = zip(*anchors)
        data, target = torch.cat(data), torch.cat(target)
        anchor_loader = InfiniteDataLoader(TensorDataset(data.cpu(), target.cpu()), weights=None, batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)

        optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.train.lr)
        for i, (data, target) in enumerate(anchor_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            optimizer.zero_grad()
            output = self.fc(self.encoder(data))
            # soft_target = torch.eye(output.shape[1])[target].to(self.config.device)
            # alpha = 0.2
            # soft_target = soft_target * (1 - alpha) + alpha / output.shape[1]
            loss = self.config.metric.loss_func(output, target)
            loss.backward()
            optimizer.step()
            if i > 100:
                break

    @torch.enable_grad()
    def cluster_train(self):
        # data, target, feat = zip(*self.buffer)
        # data, target, feat = torch.cat(data), torch.cat(target), np.concatenate(feat)
        # print('Kmeans')
        # kmeans = KMeans(n_clusters=self.budgets).fit(feat)
        # print('K nearest')
        # closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feat)
        data, target, _, _ = zip(*self.anchors)
        data, target = torch.cat(data), torch.cat(target)
        anchor_loader = InfiniteDataLoader(TensorDataset(data.cpu(), target.cpu()), weights=None,
                                           batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.ActiveTTA.lr, momentum=0.9)
        print('Cluster train')
        for i, (data, target) in enumerate(anchor_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            optimizer.zero_grad()
            output = self.fc(self.encoder(data))
            loss = self.config.metric.loss_func(output, target)
            loss.backward()
            optimizer.step()
            if i + 1 >= self.config.atta.ActiveTTA.steps:
                break

    # Active contrastive learning
    @torch.enable_grad()
    def adapt_on_env(self, env_id):

        with torch.no_grad():
            for data, target in tqdm(self.buffer_fast_loader[env_id]):
                data, target = data.to(self.config.device), target.to(self.config.device)
                feats = self.encoder(data).cpu().detach()
                if self.anchors:
                    num_anchors = torch.cat(list(zip(*self.anchors))[0]).shape[0]

                    anchor_data = torch.cat(list(zip(*self.anchors))[0]).to(self.config.device)
                    updated_anchor_feats = self.encoder(anchor_data).cpu().detach()

                    anchors_feats = torch.cat([updated_anchor_feats, feats])
                    sample_weight = torch.cat([torch.cat(list(zip(*self.anchors))[3]), torch.ones(target.shape[0])])
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(anchors_feats, sample_weight=sample_weight)
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, anchors_feats)
                    prev_anchor_labels = torch.zeros(len(raw_closest), dtype=torch.bool)
                    prev_anchor_labels[torch.tensor(kmeans.labels_[:num_anchors]).unique().long()] = True
                    closest = raw_closest[~ prev_anchor_labels] - num_anchors
                    weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1][~ prev_anchor_labels]
                else:
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(feats)
                    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
                    weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1]
                self.budgets -= len(closest)
                self.anchors.append((data[closest], target[closest], feats[closest], weights))
                self.n_clusters += 1
                if self.cold_start > 0:
                    self.cold_start -= 1
                else:
                    self.cluster_train()
        # self.cluster_train()
        # budget = 100
        # buf_size = 10
        # preheat = 100
        # class_anchors = [[] for _ in range(self.config.dataset.num_classes)]
        # shift_anchors = []
        # contrast_anchors = []
        # augmentation_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        #                                               transforms.RandomGrayscale()])
        # buffers = []
        # anchors = []
        # loss = []
        # acc = []
        # new_anchor = 0
        # for i, (data, target) in enumerate(self.single_fast_loader[env_id]):
        #     data, target = data.to(self.config.device), target.to(self.config.device)
        #     feat = self.encoder(data)
        #     output = self.fc(feat)
        #     pseudo_label = output.softmax(1).argmax(1)
        #     entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
        #     buffers.append((data, feat, pseudo_label, entropy, target))
        #     update = False
        #     if i == preheat:
        #         affine_shift_points, budget, buffers, class_anchors, shift_anchors = self.first_select(
        #             budget, buffers, class_anchors, shift_anchors)
        #         update = True
        #     elif i > preheat:
        #         affine_shift_points = []
        #         if len(buffers) >= buf_size and budget > 0:
        #             affine_shift_points, budget, buffers, class_anchors, shift_anchors = self.incremental_select(
        #                 affine_shift_points, budget, buffers, class_anchors, shift_anchors)
        #             update = True
        #         else:
        #             affine_shift_points = []
        #     if i >= preheat and update:
        #         class_anchors = self.update_class_anchors(class_anchors)
        #         shift_anchors = self.update_shift_anchors(shift_anchors)
        #         test_train_samples = list(itertools.chain(*class_anchors, shift_anchors))
        #         test_train_entropys = list(zip(*test_train_samples))[3]
        #         uncertain_samples = [sample for sample, entropy in zip(test_train_samples, test_train_entropys) if entropy > self.config.atta.ActiveTTA.eh]
        #         test_train_rounds = 1
        #         while len(uncertain_samples) > 0 and test_train_rounds > 0:
        #             self.test_time_train(test_train_samples)
        #
        #             class_anchors = self.update_class_anchors(class_anchors)
        #             shift_anchors = self.update_shift_anchors(shift_anchors)
        #             test_train_samples = list(itertools.chain(*class_anchors, shift_anchors))
        #             test_train_entropys = list(zip(*test_train_samples))[3]
        #             uncertain_samples = [sample for sample, entropy in zip(test_train_samples, test_train_entropys) if
        #                                  entropy > self.config.atta.ActiveTTA.eh]
        #             test_train_rounds -= 1
        #         unsolved_shift = self.check_affine_shift_points(affine_shift_points)
        #         shift_anchors = shift_anchors + unsolved_shift
        #         budget -= len(unsolved_shift)
        #
        #
        #     output = self.fc(feat)
        #     loss.append(self.config.metric.loss_func(output, target).detach().item())
        #     acc.append(self.config.metric.score_func(target, output))
        #     print(f'Env {env_id} Adapt set {i}: Average loss: {np.mean(loss):.4f}, Accuracy: {np.mean(acc):.4f}, Budget: {budget}, Class anchors: {len(class_anchors[0])}, {len(class_anchors[1])}, {len(class_anchors[2])}, {len(class_anchors[3])}, {len(class_anchors[4])}, {len(class_anchors[5])}, {len(class_anchors[6])}, Shift anchors: {len(shift_anchors)}')

    def incremental_select(self, affine_shift_points, budget, buffers, class_anchors, shift_anchors):
        data, feat, pseudo_label, entropy, target = zip(*buffers)
        pseudo_label = torch.cat(pseudo_label)
        entropy = torch.tensor(entropy)
        entropy_rank = torch.argsort(entropy)
        high_entropy_idx = entropy_rank[-2:]
        low_entropy_idx = torch.where(entropy < self.config.atta.ActiveTTA.el)[0]
        # for high entropy samples, they are largely misclassified, so we need to shift them to the right class
        shift_anchors_idx = np.intersect1d(high_entropy_idx.cpu(),
                                           torch.where(entropy > self.config.atta.ActiveTTA.eh)[0].cpu())
        shift_anchors_idx, affine_shift_idx = self.diversely_select(shift_anchors_idx, feat,
                                                                    self.config.dataset.num_classes)
        # for low entropy samples, they are potentially misclassified, we only consider the feature-identifiable ones
        negative_class_anchors_idx = []
        for cls in range(self.config.dataset.num_classes):
            class_feat = torch.cat(list(zip(*class_anchors[cls]))[1])
            class_mean = class_feat.mean(0)
            class_max_dis = []
            for fe in class_feat:
                class_max_dis.append(self.feature_distance(fe, class_mean, dis_type='cos'))
            class_max_dis = torch.stack(class_max_dis).max()
            cls_low_entropy_idx = np.intersect1d(low_entropy_idx.cpu(), torch.where(pseudo_label == cls)[0].cpu())
            for idx in cls_low_entropy_idx:
                if self.feature_distance(feat[idx].squeeze(), class_mean, dis_type='cos') > class_max_dis:
                    negative_class_anchors_idx.append(idx)
                    class_anchors[cls].append(buffers[idx])
                    break
        # trust_samples_idx = np.nonzero((entropy > self.config.atta.ActiveTTA.el) & (entropy < self.config.atta.ActiveTTA.eh))
        shift_anchors = shift_anchors + [buffers[idx] for idx in shift_anchors_idx]
        affine_shift_points = [buffers[idx] for idx in affine_shift_idx]
        # trust_samples = buffers[trust_samples_idx]
        buffers = []
        budget -= len(np.unique(list(itertools.chain(negative_class_anchors_idx, shift_anchors_idx))))
        return affine_shift_points, budget, buffers, class_anchors, shift_anchors

    def first_select(self, budget, buffers, class_anchors, shift_anchors):
        data, feat, pseudo_label, entropy, target = zip(*buffers)
        pseudo_label = torch.cat(pseudo_label)
        entropy = torch.tensor(entropy, device=self.config.device)
        entropy_rank = torch.argsort(entropy)
        high_entropy_idx = entropy_rank[-10:]
        low_entropy_idx = entropy_rank[:10]
        class_buffers = [torch.where(pseudo_label == cls)[0] for cls in range(self.config.dataset.num_classes)]
        class_anchors_idx = [[] for _ in range(self.config.dataset.num_classes)]
        for cls in range(self.config.dataset.num_classes):
            cls_entropy = entropy[class_buffers[cls]]
            if len(cls_entropy) > 0:
                class_anchors_idx[cls] = [class_buffers[cls][torch.argmin(cls_entropy)]]
            class_anchors_idx[cls] = torch.tensor(class_anchors_idx[cls]).numpy()
        shift_anchors_idx = np.intersect1d(high_entropy_idx.cpu(),
                                           torch.where(entropy > self.config.atta.ActiveTTA.eh)[0].cpu())
        shift_anchors_idx, affine_shift_idx = self.diversely_select(shift_anchors_idx, feat,
                                                                    self.config.dataset.num_classes)
        # trust_samples_idx = np.nonzero((entropy > self.config.atta.ActiveTTA.el) & (entropy < self.config.atta.ActiveTTA.eh))
        class_anchors = [[buffers[idx] for idx in class_anchors_idx[cls]] for cls in
                         range(self.config.dataset.num_classes)]
        shift_anchors = [buffers[idx] for idx in shift_anchors_idx]
        affine_shift_points = [buffers[idx] for idx in affine_shift_idx]
        # trust_samples = buffers[trust_samples_idx]
        buffers = []
        budget -= len(np.unique(list(itertools.chain(*class_anchors_idx, shift_anchors_idx))))
        return affine_shift_points, budget, buffers, class_anchors, shift_anchors

    @torch.no_grad()
    def naive_adapt(self, env_id):
        self.encoder.eval()
        self.fc.train()
        budget = 100
        anchors = []
        loss = []
        acc = []
        new_anchor = 0
        for i, (data, target) in enumerate(self.single_fast_loader[env_id]):
            data, target = data.to(self.config.device), target.to(self.config.device)
            feat = self.encoder(data)
            output = self.fc(feat)
            entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
            if len(anchors) < budget or new_anchor != 0:
                anchors.append((data, target, feat))
                new_anchor += 1
                if new_anchor >= 100:
                    self.naive_test_time_train(anchors)
                    new_anchor = 0
                elif len(anchors) == budget:
                    self.naive_test_time_train(anchors)
                    new_anchor = 0
            output = self.fc(feat)
            loss.append(self.config.metric.loss_func(output, target).detach().item())
            acc.append(self.config.metric.score_func(target, output))
            print(f'Env {env_id} Adapt set {i}: Average loss: {np.mean(loss):.4f}, Accuracy: {np.mean(acc):.4f}')

    @torch.no_grad()
    def extract_pretrained_feat(self, loader, batch_size):
        print('Extracting features from the loader')
        feats, targets = [], []
        for data, target in loader:
            data, target = data.to(self.config.device), target.to(self.config.device)
            feat = self.encoder(data)
            feats.append(feat.cpu())
            targets.append(target.cpu())
        feats = torch.cat(feats, dim=0)
        targets = torch.cat(targets, dim=0)
        feat_dataset = TensorDataset(feats, targets)
        weights = misc.make_weights_for_balanced_classes(
            feat_dataset) if self.config.dataset.class_balanced else None
        inf_loader = InfiniteDataLoader(dataset=feat_dataset, weights=weights,
                                        batch_size=batch_size, num_workers=self.config.num_workers)
        return inf_loader

    def diversely_select(self, idx, feat, num):
        # feat = feat[idx]
        # feat = feat.cpu().numpy()
        # kmeans = KMeans(n_clusters=num, random_state=0).fit(feat)
        # cluster_idx = kmeans.labels_
        # selected_idx = []
        # for i in range(num):
        #     selected_idx.append(idx[np.argmin(kmeans.transform(feat)[cluster_idx == i])])
        # return np.array(selected_idx), idx[cluster_idx == 0]
        return idx, np.array([])

    @torch.no_grad()
    def update_class_anchors(self, class_anchors):
        new_class_anchors = [[] for _ in range(self.config.dataset.num_classes)]
        for i, (data, _, _, _, target) in enumerate(itertools.chain(*class_anchors)):
            feat = self.encoder(data)
            output = self.fc(feat)
            pseudo_label = output.softmax(1).argmax(1)
            entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
            new_class_anchors[target].append((data, feat, pseudo_label, entropy, target))
        return new_class_anchors

    @torch.no_grad()
    def update_shift_anchors(self, shift_anchors):
        new_shift_anchors = []
        for data, feat, pseudo_label, entropy, target in shift_anchors:
            output = self.fc(feat)
            pseudo_label = output.softmax(1).argmax(1)
            entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
            new_shift_anchors.append((data, feat, pseudo_label, entropy, target))
        return new_shift_anchors

    @torch.no_grad()
    def check_affine_shift_points(self, affine_shift_points):
        unsolved_shift = []
        for data, feat, pseudo_label, entropy, target in affine_shift_points:
            output = self.fc(self.encoder(data))
            entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
            if entropy > self.config.atta.ActiveTTA.eh:
                unsolved_shift.append((data, feat, pseudo_label, entropy, target))
        return unsolved_shift

    @torch.no_grad()
    def check_and_update_anchors(self, anchors):
        new_class_anchors = [[] for _ in range(self.config.dataset.num_classes)]
        for i, (data, _, _, _, target) in enumerate(zip(*anchors)):
            feat = self.encoder(data)
            output = self.fc(feat)
            pseudo_label = output.softmax(1).argmax(1)
            entropy = -(output.softmax(1) * output.log_softmax(1)).sum().item()
            new_class_anchors[target] = (data, feat, pseudo_label, entropy, target)
        return new_class_anchors
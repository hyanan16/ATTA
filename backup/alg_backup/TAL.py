from typing import Dict
from typing import Union

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torch.utils.data import DataLoader

from GTTA.ood_algorithms.BaseOOD import BaseOODAlg
from GTTA.utils.config_reader import Conf
from GTTA.utils.register import register
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
# import models for resnet18
from torchvision.models import resnet18
import itertools
import os
import GTTA.data.loaders.misc as misc
from GTTA import register
from GTTA.utils.config_reader import Conf
from GTTA.utils.config_reader import Conf
from GTTA.utils.initial import reset_random_seed
from GTTA.utils.initial import reset_random_seed
from GTTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset, Subset


@register.alg_register
class TAL:
    def __init__(self, config: Conf):
        super(TAL, self).__init__()
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
        self.single_fast_loader = [FastDataLoader(env, weights=None,
                                                  batch_size=1,
                                                  num_workers=self.config.num_workers) for env in self.dataset]

        self.encoder = register.models[config.model.name](config).to(self.config.device)
        self.fc = nn.Linear(self.encoder.n_outputs, config.dataset.num_classes).to(self.config.device)
        self.model = nn.Sequential(self.encoder, self.fc).to(self.config.device)
        self.budgets = self.config.atta.budgets
        self.anchors = []


        if not os.path.exists(self.config.ckpt_dir):
            os.makedirs(self.config.ckpt_dir)

    def __call__(self, *args, **kwargs):
        self.train_on_env(0, train_only_fc=False, train_or_load='load')
        self.test_on_env(0)
        self.test_on_env(1)
        self.test_on_env(2)
        self.test_on_env(3)
        self.adapt_on_env(1)
        self.test_on_env(0)
        self.test_on_env(1)
        self.test_on_env(2)
        self.test_on_env(3)
        self.adapt_on_env(2)
        self.test_on_env(0)
        self.test_on_env(1)
        self.test_on_env(2)
        self.test_on_env(3)
        self.adapt_on_env(3)
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

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if x.shape[1] == 1:
            x = torch.cat([x, -x], dim=1)
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    # Active contrastive learning
    @torch.enable_grad()
    def adapt_on_env(self, env_id):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.TAL.lr, momentum=0.9)

        new_anchors = False
        for data, target in self.fast_loader[env_id]:
            data, target = data.to(self.config.device), target.to(self.config.device)
            outputs = self.model(data)
            entropies = self.softmax_entropy(outputs)
            anchor_idx = torch.where(entropies > self.config.atta.TAL.e0)[0]
            if len(anchor_idx) != 0 and self.budgets > 0:
                if len(anchor_idx) > self.budgets:
                    anchor_idx = anchor_idx[:self.budgets]
                self.anchors.append((data[anchor_idx], target[anchor_idx]))
                self.budgets -= len(anchor_idx)
                new_anchors = True
        if new_anchors:
            data, target = zip(*self.anchors)
            data, target = torch.cat(data), torch.cat(target)
            print(f'Train anchors, anchors: {len(data)}')
            anchor_loader = InfiniteDataLoader(TensorDataset(data.cpu(), target.cpu()), weights=None,
                                               batch_size=self.config.train.train_bs,
                                               num_workers=self.config.num_workers)
            for i, (data, target) in enumerate(anchor_loader):
                data, target = data.to(self.config.device), target.to(self.config.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.config.metric.loss_func(output, target)
                loss.backward()
                optimizer.step()
                if i > self.config.atta.TAL.steps:
                    break


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

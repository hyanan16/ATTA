r"""Training pipeline: training/evaluation structure, batch training.
"""
from typing import Dict
from typing import Union

import numpy as np
from torch.utils.data import DataLoader

from GTTA.ood_algorithms.BaseOOD import BaseOODAlg
from backup.tta_algorithms.tent import Tent
from torch import nn
import torch
from torch.distributions import Normal
# import models for resnet18
import os
import GTTA.data.loaders.misc as misc
from GTTA import register
from GTTA.utils.config_reader import Conf
from GTTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset


@register.tta_register
class TICP(Tent):
    r"""
    Kernel pipeline.
    This is an algorithm:
    Step1: I want to learn a variantional auto encoder to get rich features. And record the feature distributions like mean and standard deviations.
    Step2: Randomly drop features N times to mimic N pseudo environments. Then in each environment, I train a simple linear classifier.
    Step3: In test phase when I can only access test data without access to training data, I match the test data feature to the feature distributions (from step1), then filter those "out-of-distribution features” through the matching. Therefore, I get a feature mask where some features are filtered out. I match this feature mask with the pseudo environments from Step2, and get several similar pseudo environments.
    Step4: I use those classifiers of the similar pseudo environments to make an average emsemble prediction.
    Create classes for BetaVAE and linear classifiers
    The main class is the algorithm class

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Conf): Please refer to :ref:`configs:GTTA Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
                 config: Conf):

        super(TICP, self).__init__(task, model, loader, ood_algorithm, config)
        self.task: str = task

        self.dataset = register.datasets[config.dataset.name](config.dataset.dataset_root, config.dataset.test_envs,
                                                              config)
        config.dataset.dataset_type = 'image'
        config.dataset.input_shape = self.dataset.input_shape
        config.dataset.num_classes = 1 if self.dataset.num_classes == 2 else self.dataset.num_classes
        config.model.model_level = 'image'
        config.metric.set_score_func(self.dataset.metric)
        config.metric.set_loss_func(self.dataset.task)

        self.inf_loader = [InfiniteDataLoader(env, weights=None, batch_size=self.config.train.train_bs, num_workers=self.config.num_workers) for env in self.dataset]
        self.fast_loader = [FastDataLoader(env, weights=None,
                                              batch_size=self.config.train.train_bs,
                                              num_workers=self.config.num_workers) for env in self.dataset]
        self.single_fast_loader = [FastDataLoader(env, weights=None,
                                           batch_size=1,
                                           num_workers=self.config.num_workers) for env in self.dataset]

        self.encoder = register.models[config.model.name](config).to(self.config.device)
        self.fc = nn.Linear(self.encoder.n_outputs, config.dataset.num_classes).to(self.config.device)
        # self.fc = nn.utils.weight_norm(nn.Linear(self.encoder.n_outputs, config.dataset.num_classes, bias=False), dim=0).to(self.config.device)
        # self.model = nn.Sequential(model, nn.Linear(model.n_outputs, config.dataset.num_classes))
        self.vae: BetaVAE = BetaVAE(config)
        # self.model = nn.Sequential(self.vae.encoder.resnet, nn.Linear(model.n_outputs, config.dataset.num_classes))
        # self.full_feature_classifier: nn.Module = nn.Linear(config.model.dim_hidden, config.dataset.num_classes).to(self.config.device)
        self.full_feature_classifier: nn.Module = nn.Linear(config.model.dim_hidden, config.dataset.num_classes, bias=False).to(self.config.device)
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.optimizer: torch.optim.Optimizer = None
        self.episodic = config.tta.episodic
        self.config: Conf = config
        self.pseudo_environments = None
        self.mu_all, self.logvar_all = None, None
        # If the ckpt dir is not existed, create it
        if not os.path.exists(self.config.ckpt_dir):
            os.makedirs(self.config.ckpt_dir)



    def __call__(self):
        if self.task in ['train', 'test']:
            # # self.train()
            # train_or_load = self.config.tta.TICP.train_or_load
            # self.train_vae(train_or_load)
            # self.mu_all, self.logvar_all = self.record_feature_distributions(train_or_load)
            # self.pseudo_environments = self.create_pseudo_environments(train_or_load)
            # self.train_linear_classifiers(self.pseudo_environments, train_or_load)
            # # self.train_linear_classifier_on_test(self.pseudo_environments, 'train')
            #
            # self.adapt('test')
            self.train_on_env(0, train_only_fc=False, train_or_load='load')
            self.test_on_env(0)
            # # self.train_on_env(1, train_only_fc = True)
            self.test_on_env(1)
            # # self.train_on_env(2, train_only_fc = True)
            self.test_on_env(2)
            # # self.train_on_env(3, train_only_fc = True)
            self.test_on_env(3)
            feats, targets = self.record_distribution(0, 'load')
            feat_mu = feats.mean(0).to(self.config.device)
            feat_std = feats.std(0).to(self.config.device)
            # base_mask = self.filter_base_mask(0, feat_mu, 'load')
            # masks = self.train_dropout_classifiers(0, feat_mu, base_mask, 'train')
            # # self.test_on_env_with_global_dropout(0, (feat_mu, feat_std), masks)
            # # self.test_on_env_with_global_dropout(1, (feat_mu, feat_std), masks)
            # # self.test_on_env_with_global_dropout(2, (feat_mu, feat_std), masks)
            # self.test_on_env_with_global_dropout(3, (feat_mu, feat_std), masks)
            self.test_on_env_with_full_hole_covered(3, (feat_mu, feat_std))

        elif self.task == 'adapt':
            print('#D#Config model')
            self.config_model('test', load_param=True)
            self.configure_model()
            params, param_names = self.collect_params()
            print(f'#I#{param_names}')
            self.optimizer = SAM(params, torch.optim.SGD, lr=self.config.tta.SAR.lr, momentum=0.9)

            self.model_state, self.optimizer_state = \
                self.copy_model_and_optimizer()

            self.adapt('test')

    def filter_base_mask(self, env_id, feat_mu, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/base_masks.npy'):
            self.encoder.train()
            self.fc.train()
            base_mask = nn.Parameter(torch.zeros(512).to(self.config.device), requires_grad=True)
            optimizer = torch.optim.Adam(list(self.fc.parameters()) + [base_mask], lr=self.config.train.lr)
            inf_loader = self.extract_pretrained_feat(self.fast_loader[env_id], self.config.train.train_bs)
            for batch_idx, (data, target) in enumerate(inf_loader):
                optimizer.zero_grad()
                data, target = data.to(self.config.device), target.to(self.config.device)
                att = self.concrete_sample(base_mask, 1, True)
                output = self.fc(data * att + feat_mu * (1 - att))
                eps = 1e-6
                r = 0.5
                info_loss = (att * torch.log(att / r + eps) +
                             (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()
                loss = self.config.metric.loss_func(output, target) + 1000 * info_loss
                acc = self.config.metric.score_func(target, output)
                loss.backward()
                optimizer.step()
                if batch_idx % self.config.train.log_interval == 0:
                    print(f'Iteration: {batch_idx} Loss: {loss.item():.4f} Acc: {acc:.4f} att: {att.sum():.4f}')
                if batch_idx > self.config.train.max_iters:
                    break
            print((self.concrete_sample(base_mask, 1, False) > 0.5))
            base_mask = (self.concrete_sample(base_mask, 1, False) > 0.5).long().detach().cpu().numpy()
            np.save(self.config.ckpt_dir + '/base_masks.npy', base_mask)
        else:
            base_mask = np.load(self.config.ckpt_dir + '/base_masks.npy')
        return base_mask

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        # if training:
        # TODO: Change it back to if training
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

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
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/encoder.pth'):
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
                optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.fc.parameters()), lr=self.config.train.lr)
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
            torch.save(self.encoder.state_dict(), self.config.ckpt_dir + '/encoder.pth')
            torch.save(self.fc.state_dict(), self.config.ckpt_dir + '/fc.pth')
        else:
            self.encoder.load_state_dict(torch.load(self.config.ckpt_dir + '/encoder.pth', map_location=self.config.device))
            self.fc.load_state_dict(torch.load(self.config.ckpt_dir + '/fc.pth', map_location=self.config.device))

    @torch.no_grad()
    def test_on_env_with_global_dropout(self, env_id, orig_feats, masks):
        feat_mu, feat_std = orig_feats
        self.encoder.eval()
        self.fc.eval()
        test_loss = 0
        test_acc = 0
        masks = torch.from_numpy(masks).float().to(self.config.device)
        hole_cover_rate = []
        for data, target in self.fast_loader[env_id]:
            data, target = data.to(self.config.device), target.to(self.config.device)

            feat = self.encoder(data)
            attention = ((feat < (feat_mu + 2 * feat_std)) & (feat > (feat_mu - 2 * feat_std))).float()
            similarity = torch.matmul(attention, masks.t())
            hole_cover_rate.append((torch.matmul(1 - attention, (1 - masks).t()) / (1 - attention + 1e-6).sum(1, keepdim=True)).max(1)[0].mean().item())
            normalized_similarity = (similarity - similarity.min(1, keepdim=True)[0]) / (similarity.max(1, keepdim=True)[0] - similarity.min(1, keepdim=True)[0]) * 10
            similarity = normalized_similarity.exp() / torch.sum(normalized_similarity.exp(), 1, keepdim=True)
            # calculate the weighted average of the classifiers as the final prediction
            outputs = torch.zeros(data.shape[0], self.config.dataset.num_classes).to(self.config.device)
            for i in range(self.config.tta.TICP.num_drops):
                outputs += similarity[:, i].unsqueeze(1) * self.classifier[i](feat * masks[i] + (1 - masks[i]) * feat_mu)
            # outputs = self.classifier[0](z)

            test_loss += self.config.metric.loss_func(outputs, target, reduction='sum').item()
            test_acc += self.config.metric.score_func(target, outputs) * len(data)
        test_loss /= len(self.fast_loader[env_id].dataset)
        test_acc /= len(self.fast_loader[env_id].dataset)
        print(f'Env {env_id} Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        print(f'Env {env_id} Hole cover rate: {np.mean(hole_cover_rate):.4f}')

    def fine_tune_full_hole(self, inf_loader, attention, feat_mu):
        optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.tta.TICP.classifier_lr)
        avg_loss = []
        avg_acc = []
        for fine_tune_id, (data, target) in enumerate(inf_loader):
            optimizer.zero_grad()
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.fc(data * attention)
            loss = self.config.metric.loss_func(output, target)
            acc = self.config.metric.score_func(target, output)
            loss.backward()
            optimizer.step()
            avg_loss += [loss.item()]
            avg_acc += [acc]
            # if fine_tune_id % self.config.train.log_interval == 0:
            #     print(
            #         f'Iteration: {fine_tune_id} Loss: {np.mean(avg_loss[-100:]):.4f} Acc: {np.mean(avg_acc[-100:]):.4f}')
            if fine_tune_id > 100:
                break

    @torch.enable_grad()
    def test_on_env_with_full_hole_covered(self, env_id, orig_feats):
        feat_mu, feat_std = orig_feats
        self.encoder.eval()
        self.fc.eval()
        test_loss = []
        test_acc = []
        hole_cover_rate = []
        entropys = []
        attentions = []
        inf_loader = self.extract_pretrained_feat(self.fast_loader[0], self.config.train.train_bs)
        import copy
        orig_state_dict = copy.deepcopy(self.fc.state_dict())
        for i, (data, target) in enumerate(self.single_fast_loader[env_id]):
            data, target = data.to(self.config.device), target.to(self.config.device)

            feat = self.encoder(data)
            attention = ((feat < (feat_mu + 4 * feat_std)) & (feat > (feat_mu - 4 * feat_std))).float()
            # use this attention to fine-tune the classifier
            # self.fc.load_state_dict(orig_state_dict)
            # self.fine_tune_full_hole(inf_loader, attention, feat_mu)

            outputs = self.fc(feat)
            attentions.append(512 - attention.sum().item())
            entropys.append(-(outputs.softmax(1) * outputs.log_softmax(1)).sum().item())

            test_loss.append(self.config.metric.loss_func(outputs, target, reduction='sum').item())
            test_acc.append(self.config.metric.score_func(target, outputs) * len(data))
            print(f'Env {env_id} Test set {i}: Average loss: {np.mean(test_loss):.4f}, Accuracy: {np.mean(test_acc):.4f}')
            print(list(zip(attentions, entropys, test_acc)))
        print(f'Env {env_id} Test set: Average loss: {np.mean(test_loss):.4f}, Accuracy: {np.mean(test_acc):.4f}')
        print(f'Env {env_id} Hole cover rate: {hole_cover_rate}, {np.mean(hole_cover_rate):.4f}')


    @torch.no_grad()
    def record_distribution(self, env_id, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/feats.pth'):
            self.encoder.eval()
            self.fc.eval()
            feats, targets = [], []
            for data, target in self.fast_loader[env_id]:
                data, target = data.to(self.config.device), target.to(self.config.device)
                feat = self.encoder(data)
                feats.append(feat.cpu())
                targets.append(target.cpu())
            feats = torch.cat(feats, dim=0)
            targets = torch.cat(targets, dim=0)
            torch.save((feats, targets), self.config.ckpt_dir + '/feats.pth')
        else:
            feats, targets = torch.load(self.config.ckpt_dir + '/feats.pth', map_location=self.config.device)
        return feats, targets


    def train_dropout_classifiers(self, env_id, feat_mu, base_mask, train_or_load):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/masks.npy'):
            # create pseudo environments
            masks = []
            for i in range(self.config.tta.TICP.num_drops):
                mask = np.random.choice([0, 1], size=512, p=[0.1, 0.9])
                mask = mask * base_mask
                masks.append(mask)

            # save the masks
            masks = np.array(masks)
            np.save(self.config.ckpt_dir + '/masks.npy', masks)
        else:
            masks = np.load(self.config.ckpt_dir + '/masks.npy')

        # train the classifiers
        if train_or_load == 'train' or not os.path.exists(
                self.config.ckpt_dir + f'/linear_classifier_{self.config.tta.TICP.num_drops - 1}.pt'):
            # train linear classifiers
            self.encoder.eval()
            self.classifier = []
            inf_loader = self.extract_pretrained_feat(self.fast_loader[env_id], self.config.train.train_bs)

            for i in range(self.config.tta.TICP.num_drops):
                print('Training linear classifier for mask: ', i)
                mask = torch.tensor(masks[i]).to(self.config.device)
                # train linear classifier
                classifier = nn.Linear(512, self.config.dataset.num_classes).to(
                    self.config.device)
                # classifier.load_state_dict(self.fc.state_dict())
                classifier.train()
                optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.tta.TICP.classifier_lr)
                avg_loss = []
                avg_acc = []
                for batch_idx, (data, target) in enumerate(inf_loader):
                    optimizer.zero_grad()
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    output = classifier(data * mask + (1 - mask) * feat_mu)
                    loss = self.config.metric.loss_func(output, target)
                    acc = self.config.metric.score_func(target, output)
                    loss.backward()
                    optimizer.step()
                    avg_loss += [loss.item()]
                    avg_acc += [acc]
                    if batch_idx % self.config.train.log_interval == 0:
                        print(f'Iteration: {batch_idx} Loss: {np.mean(avg_loss[-100:]):.4f} Acc: {np.mean(avg_acc[-100:]):.4f}')
                    if batch_idx > self.config.tta.TICP.classifier_epochs:
                        break

                # save the trained linear classifiers for later test-time adaptation
                torch.save(classifier.state_dict(), self.config.ckpt_dir + '/linear_classifier_' + str(i) + '.pt')
                classifier.eval()
                self.classifier.append(classifier)
        else:
            self.classifier = []
            for i in range(self.config.tta.TICP.num_drops):
                classifier = nn.Linear(512, self.config.dataset.num_classes).to(
                    self.config.device)
                classifier.load_state_dict(torch.load(self.config.ckpt_dir + f'/linear_classifier_{i}.pt'))
                classifier.eval()
                self.classifier.append(classifier)
        return masks

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

    def trainable_parameters(self, train_batchnorm=True):
        encoder_batchnorm_params = []
        for m in self.vae.encoder.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                encoder_batchnorm_params += list(m.parameters())
        vae_fc_params = list(self.vae.encoder.fc.parameters())
        vae_decoder_params = list(self.vae.decoder.parameters())
        classifier_params = list(self.full_feature_classifier.parameters())
        if train_batchnorm:
            return vae_fc_params + vae_decoder_params + classifier_params + encoder_batchnorm_params
        else:
            return vae_fc_params + vae_decoder_params + classifier_params

    # train the BetaVAE model
    def train_vae(self, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/vae_model.pt'):
            self.config_model(self.task, load_param=False)
            # vae_optimizer = torch.optim.Adam(list(self.vae.encoder.fc.parameters()) + list(self.vae.decoder.parameters()), lr=self.config.train.lr)
            cls_optimizer = torch.optim.Adam(list(self.vae.encoder.parameters()) + list(self.full_feature_classifier.parameters()), lr=self.config.train.lr)
            resnet_optimizer = torch.optim.Adam(self.vae.encoder.resnet.parameters(), lr=self.config.train.lr)
            whole_optimizer = torch.optim.Adam(list(self.vae.parameters()) + list(self.full_feature_classifier.parameters()), lr=self.config.train.lr)
            second_optimizer = torch.optim.Adam(list(self.vae.encoder.fc.parameters()) + list(self.vae.decoder.parameters()) + list(self.full_feature_classifier.parameters()), lr=self.config.train.lr)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr)
            # self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.config.train.lr)
            self.vae.train()
            self.full_feature_classifier.train()
            # self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.config.train.lr)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.train.mile_stones,
            #                                                       gamma=self.config.train.gamma)
            loader = self.fast_loader[0]

            for epoch in range(self.config.train.max_epoch):
                acc = 0
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    if len(data.shape) == 4:
                        recon_batch, mu, logvar, z, feat = self.vae(data)
                    else:
                        recon_batch, mu, logvar, z = self.vae(data)
                    # How about detach the mu for classifier?
                    output = self.full_feature_classifier(mu)
                    cls_loss = self.config.metric.loss_func(output, target)


                    cls_optimizer.zero_grad()
                    cls_loss.backward()
                    cls_optimizer.step()
                    # loss = cls_loss
                    # loss.backward()
                    if self.config.dataset.name in ['VLCS', 'PACS']:
                        acc += self.config.metric.score_func(target, output)
                    else:
                        acc += self.config.metric.score_func(target, output.sigmoid())
                    if batch_idx % self.config.train.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx, len(loader),
                                   100. * batch_idx / len(loader),
                                   cls_loss.item()))
                # self.scheduler.step()
                print('Train Epoch: {} Average accuracy: {:.6f}'.format(epoch, acc / len(loader)))
                # print the two losses
                print('classification loss', cls_loss.item())

            with torch.no_grad():
                print('Extracting features from the train set')
                feats, targets = [], []
                for data, target in loader:
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    feat = self.vae.encoder.resnet(data)
                    feats.append(feat.cpu())
                    targets.append(target.cpu())
                feats = torch.cat(feats, dim=0)
                targets = torch.cat(targets, dim=0)
                feat_dataset = TensorDataset(feats, targets)
                weights = misc.make_weights_for_balanced_classes(
                    feat_dataset) if self.config.dataset.class_balanced else None
                feat_loader = FastDataLoader(dataset=feat_dataset, weights=weights,
                                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)


            for epoch in range(self.config.train.max_epoch):
                acc = 0
                for batch_idx, (data, target) in enumerate(feat_loader):
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    if len(data.shape) == 4:
                        recon_batch, mu, logvar, z, feat = self.vae(data)
                        vae_loss, reconstruction_loss, kl_loss = self.vae.loss_function(recon_batch, feat, mu, logvar, z)
                    else:
                        recon_batch, mu, logvar, z = self.vae(data)
                        vae_loss, reconstruction_loss, kl_loss = self.vae.loss_function(recon_batch, data, mu, logvar, z)
                    # mu, logvar, z = self.vae.get_posterior_z(data)
                    # output = self.vae.encoder(data)
                    # output = self.model(data)

                    # How about detach the mu for classifier?
                    output = self.full_feature_classifier(mu)
                    cls_loss = self.config.metric.loss_func(output, target)

                    loss = vae_loss + cls_loss

                    second_optimizer.zero_grad()
                    loss.backward()
                    second_optimizer.step()
                    # loss = cls_loss
                    # loss.backward()
                    if self.config.dataset.name in ['VLCS', 'PACS']:
                        acc += self.config.metric.score_func(target, output)
                    else:
                        acc += self.config.metric.score_func(target, output.sigmoid())
                    if batch_idx % self.config.train.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx, len(loader),
                                   100. * batch_idx / len(loader),
                                   loss.item()))
                # self.scheduler.step()
                print('Train Epoch: {} Average accuracy: {:.6f}'.format(epoch, acc / len(loader)))
                # print the two losses
                print('classification loss', cls_loss.item())
                print('reconstruction loss: ', reconstruction_loss.item(), 'kl loss: ', kl_loss.item())
                # print(recon_batch.max())
            # save the vae model and the recorded feature distributions for later test-time adaptation
            torch.save(self.vae.state_dict(), self.config.ckpt_dir + '/vae_model.pt')
        elif train_or_load == 'load':
            self.config_model(self.task, load_param=False)
            self.vae.load_state_dict(torch.load(self.config.ckpt_dir + '/vae_model.pt'))
        else:
            raise ValueError('train_or_load should be either "train" or "load"')


    # Run an epoch to record the feature distributions
    def record_feature_distributions(self, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/mu_all.pt'):
            self.vae.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.fast_loader[0]):
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    if len(data.shape) == 4:
                        recon_batch, mu, logvar, z, feat = self.vae(data)
                    else:
                        recon_batch, mu, logvar, z = self.vae(data)
                    if batch_idx == 0:
                        mu_all = mu
                        logvar_all = logvar
                        from GTTA.definitions import STORAGE_DIR
                        reconstruction_path = STORAGE_DIR + f'/reconstruction/{self.config.dataset.name}/{self.config.tta.TICP.beta}_{self.config.train.max_epoch}'
                        self.save_image(recon_batch, reconstruction_path, 'recon')
                        self.save_image(data, reconstruction_path, 'original')
                    else:
                        mu_all = torch.cat((mu_all, mu), 0)
                        logvar_all = torch.cat((logvar_all, logvar), 0)
            torch.save(mu_all, self.config.ckpt_dir + '/mu_all.pt')
            torch.save(logvar_all, self.config.ckpt_dir + '/logvar_all.pt')
        elif train_or_load == 'load':
            mu_all = torch.load(self.config.ckpt_dir + '/mu_all.pt')
            logvar_all = torch.load(self.config.ckpt_dir + '/logvar_all.pt')
        else:
            raise ValueError('train_or_load should be either "train" or "load"')
        return mu_all, logvar_all

    # Save a batch of images (batchsize * 2 * 28 * 28) into a folder
    def save_image(self, data, path, batch_idx):
        if not os.path.exists(path):
            os.makedirs(path)
        import matplotlib.pyplot as plt
        img = data.cpu().numpy()
        # expand the batchsize * 2 * 28 * 28 image into batchsize * 3 * 28 * 28 where the 3rd channel is all 0s
        if 'MNIST_CNN' in self.config.model.name:
            img = np.concatenate((img, np.zeros((img.shape[0], 1, img.shape[2], img.shape[3]))), axis=1)
        if self.config.dataset.name in ['VLCS', 'PACS']:
            img = img * np.array([0.229, 0.224, 0.225])[None, :, None, None] + np.array([0.485, 0.456, 0.406])[None, :, None, None]
            img = img.clip(0, 1)
        img = np.transpose(img, (0, 2, 3, 1))
        for i in range(img.shape[0]):
            plt.imsave(f'{path}/img_{batch_idx}_{i}.png', img[i])


    # create pseudo environments by dropping features N times RANDOMLY.
    def create_pseudo_environments(self, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + '/masks.npy'):
            # create pseudo environments
            masks = []
            for i in range(self.config.tta.TICP.num_drops):
                if i == 0:
                    masks.append(np.ones(self.config.model.dim_hidden, dtype=np.int))
                else:
                    masks.append(np.random.choice([0, 1], size=self.config.model.dim_hidden, p=[0.5, 0.5]))

            # save the masks
            masks = np.array(masks)
            np.save(self.config.ckpt_dir + '/masks.npy', masks)
        else:
            masks = np.load(self.config.ckpt_dir + '/masks.npy')
        return masks

    # Train a linear classifier per pseudo environment. These linear classifiers and environments are trained parallel with Pytorch and GPUs.
    def train_linear_classifiers(self, masks, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + f'/linear_classifier_{self.config.tta.TICP.num_drops - 1}.pt'):
            # train linear classifiers
            self.vae.eval()
            self.classifier = []

            loader = self.fast_loader[0]

            with torch.no_grad():
                print('Extracting features from the train set')
                feats, targets = [], []
                for data, target in loader:
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    feat = self.vae.encoder.resnet(data)
                    feats.append(feat.cpu())
                    targets.append(target.cpu())
                feats = torch.cat(feats, dim=0)
                targets = torch.cat(targets, dim=0)
                feat_dataset = TensorDataset(feats, targets)
                weights = misc.make_weights_for_balanced_classes(feat_dataset) if self.config.dataset.class_balanced else None
                feat_loader = FastDataLoader(dataset=feat_dataset, weights=weights, batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)

            for i in range(self.config.tta.TICP.num_drops):
                print('Training linear classifier for mask: ', i)
                mask = torch.tensor(masks[i]).to(self.config.device)
                # train linear classifier
                classifier = nn.Linear(self.config.model.dim_hidden, self.config.dataset.num_classes, bias=False).to(self.config.device)
                classifier.train()
                optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.tta.TICP.classifier_lr)
                for epoch in range(self.config.tta.TICP.classifier_epochs):
                    acc = 0
                    for batch_idx, (data, target) in enumerate(feat_loader):
                        data, target = data.to(self.config.device), target.to(self.config.device)
                        optimizer.zero_grad()
                        mu, logvar, z = self.vae.get_posterior_z(data)
                        output = classifier(z * mask)
                        loss = self.config.metric.loss_func(output, target)
                        if self.config.dataset.name in ['VLCS', 'PACS']:
                            acc += self.config.metric.score_func(target.cpu(), output.softmax(1).clone().detach().cpu())
                        else:
                            acc += self.config.metric.score_func(target.cpu(), output.sigmoid().clone().detach().cpu())
                        loss.backward()
                        optimizer.step()
                        if batch_idx % self.config.train.log_interval == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx, len(loader),
                                       100. * batch_idx / len(loader),
                                       loss.item() / len(data)))
                    print('Train Epoch: {} Average accuracy: {:.6f}'.format(epoch, acc / len(loader)))

                # save the trained linear classifiers for later test-time adaptation
                torch.save(classifier.state_dict(), self.config.ckpt_dir + '/linear_classifier_' + str(i) + '.pt')
                classifier.eval()
                self.classifier.append(classifier)
        else:
            self.classifier = []
            for i in range(self.config.tta.TICP.num_drops):
                classifier = nn.Linear(self.config.model.dim_hidden, self.config.dataset.num_classes, bias=False).to(self.config.device)
                classifier.load_state_dict(torch.load(self.config.ckpt_dir + f'/linear_classifier_{i}.pt'))
                classifier.eval()
                self.classifier.append(classifier)

    def train_linear_classifier_on_test(self, masks, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + f'/linear_classifier_{self.config.tta.TICP.num_drops - 1}.pt'):
            # train linear classifiers
            self.vae.eval()
            self.classifier = []

            with torch.no_grad():
                print('Extracting features from the test set')
                feats, targets = [], []
                for data, target in self.loader['test']:
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    feat = self.vae.encoder.resnet(data)
                    feats.append(feat.cpu())
                    targets.append(target.cpu())
                feats = torch.cat(feats, dim=0)
                targets = torch.cat(targets, dim=0)
                feat_dataset = TensorDataset(feats, targets)
                weights = misc.make_weights_for_balanced_classes(feat_dataset) if self.config.dataset.class_balanced else None
                feat_loader = FastDataLoader(dataset=feat_dataset, weights=weights, batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)

            for i in range(self.config.tta.TICP.num_drops):
                print('Training linear classifier for mask: ', i)
                mask = torch.tensor(masks[i]).to(self.config.device)
                # train linear classifier
                classifier = nn.Linear(self.config.model.dim_hidden, self.config.dataset.num_classes, bias=False).to(self.config.device)
                classifier.train()
                # batchnorm_params = []
                # for m in self.vae.encoder.fc.modules():
                #     if isinstance(m, nn.BatchNorm1d):
                #         batchnorm_params += list(m.parameters())
                # optimizer = torch.optim.Adam(list(classifier.parameters()) + batchnorm_params, lr=self.config.tta.TICP.classifier_lr)
                optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.tta.TICP.classifier_lr)
                for epoch in range(self.config.tta.TICP.classifier_epochs * 3):
                    acc = 0
                    for batch_idx, (data, target) in enumerate(feat_loader):
                        data, target = data.to(self.config.device), target.to(self.config.device)
                        optimizer.zero_grad()
                        mu, logvar, z = self.vae.get_posterior_z(data)
                        output = classifier(mu * mask)
                        loss = self.config.metric.loss_func(output, target)
                        if self.config.dataset.name in ['VLCS', 'PACS']:
                            acc += self.config.metric.score_func(target.cpu(), output.softmax(1).clone().detach().cpu())
                        else:
                            acc += self.config.metric.score_func(target.cpu(), output.sigmoid().clone().detach().cpu())
                        loss.backward()
                        optimizer.step()
                        if batch_idx % self.config.train.log_interval == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx, len(self.loader['test']),
                                       100. * batch_idx / len(self.loader['test']),
                                       loss.item() / len(data)))
                    print('Train Epoch: {} Average accuracy: {:.6f}'.format(epoch, acc / len(self.loader['test'])))

                # save the trained linear classifiers for later test-time adaptation
                # torch.save(classifier.state_dict(), self.config.ckpt_dir + '/linear_classifier_' + str(i) + '.pt')
                classifier.eval()
                self.classifier.append(classifier)
        else:
            self.classifier = []
            for i in range(self.config.tta.TICP.num_drops):
                classifier = nn.Linear(self.config.model.dim_hidden, self.config.dataset.num_classes, bias=False).to(self.config.device)
                classifier.load_state_dict(torch.load(self.config.ckpt_dir + f'/linear_classifier_{i}.pt'))
                classifier.eval()
                self.classifier.append(classifier)


    # Calculate the KL distance between the feature distribution of the current batch and the recorded feature distribution of the training set
    # Note that they are both mixtures of Gaussians
    # We will have a distance for each feature.
    # Use matrix operations.
    def calculate_distance(self, mu, logvar):
        if self.mu_all is not None and self.logvar_all is not None:
            mu_all = self.mu_all
            logvar_all = self.logvar_all
        else:
            mu_all = torch.load(self.config.ckpt_dir + '/mu_all.pt')
            logvar_all = torch.load(self.config.ckpt_dir + '/logvar_all.pt')
            mu_all = mu_all.to(self.config.device)
            logvar_all = logvar_all.to(self.config.device)
        # calculate distance
        # mu = mu.unsqueeze(1)
        # logvar = logvar.unsqueeze(1)
        # mu_all = mu_all.unsqueeze(0)
        # logvar_all = logvar_all.unsqueeze(0)
        feat_mu = mu_all.mean(0)
        feat_std = mu_all.std(0)
        # distance = kl_divergence(Normal(feat_mu, feat_std), Normal(mu, (0.5 * logvar).exp()))
        distance = ((mu < (feat_mu + feat_std)) & (mu > (feat_mu - feat_std))).float()
        # distance = torch.mean(torch.exp(logvar_all) + (mu_all - mu) ** 2 / torch.exp(logvar_all) - 1 - logvar_all, 2)
        return distance

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, data):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.vae.eval()
        self.full_feature_classifier.eval()
        # self.vae.train()
        # self.optimizer.zero_grad()
        # get features from vae
        mu, logvar, z = self.vae.get_posterior_z(data)
        # calculate distance
        distance = self.calculate_distance(mu, logvar)
        # filter out unreliable features using normalized soft attentions according to the distance
        # Smaller distance will have higher attention weights
        # We will have a weight for each feature.
        # Use matrix operations.
        # attention = torch.exp(-distance / self.config.tta.TICP.temperature)
        # attention = attention / torch.sum(attention, 1, keepdim=True)
        attention = distance

        # only 0, 5, 7 classifier works, only 1 and 4 features works

        # Using classifiers whose corresponding masks are similar to this attention
        # We will have a weight for each classifier, which is the similarity between the mask and the attention.
        # Use matrix operations.
        if self.pseudo_environments is not None:
            masks = self.pseudo_environments
        else:
            masks = np.load(self.config.ckpt_dir + '/masks.npy')
        masks = torch.from_numpy(masks).float().to(self.config.device)
        similarity = torch.matmul(attention, masks.t())
        similarity = similarity / torch.sum(similarity, 1, keepdim=True)
        # calculate the weighted average of the classifiers as the final prediction
        # outputs = torch.zeros(data.shape[0], self.config.dataset.num_classes).to(self.config.device)
        # for i in range(self.config.tta.TICP.num_drops):
        #     outputs += similarity[:, i].unsqueeze(1) * self.classifier[i](z * masks[i])
        outputs = self.classifier[0](z)
        # outputs = self.full_feature_classifier(z)

        # similarity[:, 0] = 0
        # chosen_classifier = similarity.topk(1)[1].flatten()
        # outputs = torch.zeros(data.shape[0], self.config.dataset.num_classes).to(self.config.device)
        # for i, c in enumerate(chosen_classifier):
        #     outputs[i] = self.classifier[c](z[i] * masks[c])


        reset_flag = False
        return outputs, reset_flag


    def adapt_forward(self, data):
        if self.episodic:
            self.reset()

        outputs, reset_flag = self.forward_and_adapt(data.x)
        if reset_flag:
            self.reset()

        return outputs

    def reset(self):
        super().reset()


# Several new Pytorch module classes. The main class is a Beta Variational Autoencoder (Beta-VAE) with Gaussians prior.
# The prior is a Gaussians with diagonal covariance matrices.
# The encoder is a ResNet-18 with a fully-connected layer at the end.
# The decoder is a ResNet-18 with a fully-connected layer at the end.
# The encoder, decoder, and prior should be written as separate classes.
# It is trained with a reconstruction loss and a KL-divergence loss.
# The hidden dimension is config.model.dim_hidden and the device is config.device.
class BetaVAE(nn.Module):
    def __init__(self, config):
        super(BetaVAE, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.prior = Prior(config)
        self.alpha = config.tta.TICP.alpha
        self.beta = config.tta.TICP.beta
        if 'MNIST_CNN' in config.model.name:
            self.reconstruction_loss = nn.BCELoss(reduction='none')
        else:
            self.reconstruction_loss = nn.MSELoss(reduction='none')
        self.kl_divergence_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.to(self.config.device)
        # self.init_weights()

    # # Use xaiver initialization to initialize the all the weights in the model.
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    #             nn.init.xavier_uniform_(m.weight, gain=0.1)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        if len(x.shape) == 4:
            mu, logvar, feat = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            return x_hat, mu, logvar, z, feat
        else:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            return x_hat, mu, logvar, z

    def loss_function(self, x_hat, x, mu, logvar, z):
        reconstruction_loss = self.reconstruction_loss(x_hat, x).mean(0).sum()
        kl_divergence_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0).sum()
        # kl_divergence_loss = torch.distributions.kl_divergence(Normal(mu, (0.5*logvar).exp()), self.prior()).sum(1).mean()
        loss = self.alpha * reconstruction_loss + self.beta * kl_divergence_loss
        return loss, reconstruction_loss, kl_divergence_loss

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def get_posterior_z(self, x):
        if len(x.shape) == 4:
            mu, logvar, feat = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        else:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.dim_hidden = config.model.dim_hidden
        self.num_gaussians = config.tta.TICP.num_gaussians
        self.device = config.device
        self.resnet = register.models[config.model.name](config)
        # self.fc = nn.Sequential(nn.Linear(self.resnet.n_outputs, 512), nn.BatchNorm1d(512), nn.ReLU(),
        #                         nn.Linear(self.resnet.n_outputs, 512), nn.BatchNorm1d(512), nn.ReLU(),
        #                         nn.Linear(self.resnet.n_outputs, 512), nn.BatchNorm1d(512), nn.ReLU(),
        #                         nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #                         nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        #                         nn.Linear(128, self.dim_hidden*2))
        # self.fc = nn.Sequential(nn.Linear(self.resnet.n_outputs, 512), nn.LayerNorm(512), nn.ReLU(),
        #                         nn.Linear(self.resnet.n_outputs, 512), nn.LayerNorm(512), nn.ReLU(),
        #                         nn.Linear(self.resnet.n_outputs, 512), nn.LayerNorm(512), nn.ReLU(),
        #                         nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
        #                         nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
        #                         nn.Linear(128, self.dim_hidden * 2))
        self.fc = nn.Sequential(nn.Linear(self.resnet.n_outputs, 512), nn.GroupNorm(8, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.GroupNorm(8, 512), nn.ReLU(),
                                nn.Linear(512, self.dim_hidden * 2))
        # self.fc = nn.Linear(self.resnet.n_outputs, self.dim_hidden * 2)
        # self.fc = nn.Linear(self.resnet.n_outputs, self.config.dataset.num_classes)
        self.to(self.device)

    def forward(self, x):
        if len(x.shape) == 4:
            feat = self.resnet(x)
            x = self.fc(feat)
            mu, logvar = torch.chunk(x, 2, dim=1)
            return mu, logvar, feat
        else:
            x = self.fc(x)
            mu, logvar = torch.chunk(x, 2, dim=1)
            return mu, logvar
        # return x





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.dim_hidden = config.model.dim_hidden
        self.device = config.device
        if 'MNIST_CNN' in config.model.name:
            self.fc1 = nn.Linear(self.dim_hidden, 512*3*3)
            self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1)
            self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.ConvTranspose2d(64, 2, kernel_size=5, stride=1, padding=2)
        else:
            # self.fc1 = nn.Linear(self.dim_hidden, 2048)
            # self.bn1 = nn.BatchNorm2d(2048)
            # from GTTA.networks.domainbed_networks.resnet_decoder import ResNetDecoder, Bottleneck
            # self.resnet = ResNetDecoder(Bottleneck, [3, 4, 6, 3])
            # self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            # self.bn3 = nn.BatchNorm2d(64)
            # self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
            # self.fc1 = nn.Sequential(nn.Linear(self.dim_hidden, 128), nn.BatchNorm1d(128), nn.ReLU(),
            #                          nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            #                          nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            #                          nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            #                          nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            #                          nn.Linear(512, 512))
            # self.fc1 = nn.Sequential(nn.Linear(self.dim_hidden, 128), nn.LayerNorm(128), nn.ReLU(),
            #                          nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
            #                          nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(),
            #                          nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            #                          nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            #                          nn.Linear(512, 512))
            self.fc1 = nn.Sequential(nn.Linear(self.dim_hidden, 512), nn.GroupNorm(8, 512), nn.ReLU(),
                                     nn.Linear(512, 512), nn.GroupNorm(8, 512), nn.ReLU(),
                                     nn.Linear(512, 512))
            # self.fc1 = nn.Linear(self.dim_hidden, 512)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.config = config
        self.to(self.device)

    def forward(self, x):
        if 'MNIST_CNN' in self.config.model.name:
            x = self.relu(self.fc1(x))
            x = x.view(-1, 512, 3, 3)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.sigmoid(self.conv4(x))
        else:
            # x = self.relu(self.bn1(self.fc1(x)[..., None, None]))
            # x = self.resnet(x)
            # x = self.relu(self.bn3(self.conv3(x)))
            # x = self.conv4(x)
            x = self.fc1(x)
        return x

# The prior is a mixture of Gaussians with diagonal covariance matrices.
class Prior(nn.Module):
    def __init__(self, config):
        super(Prior, self).__init__()
        self.config = config
        self.dim_hidden = config.model.dim_hidden
        self.num_gaussians = config.tta.TICP.num_gaussians
        self.device = config.device
        self.mu = nn.Parameter(torch.zeros(self.dim_hidden), requires_grad=False)
        self.logvar = nn.Parameter(torch.zeros(self.dim_hidden), requires_grad=False)
        self.to(self.device)
        std = torch.exp(0.5 * self.logvar)
        self.normal = Normal(self.mu, std)

    def forward(self):
        return self.normal

    # The log probability of a batch of samples z under the prior.

    def sample(self, num_samples):
        z = self.normal.sample((num_samples,))
        return z




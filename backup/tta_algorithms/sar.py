r"""Training pipeline: training/evaluation structure, batch training.
"""
from typing import Dict
from typing import Union

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GTTA.ood_algorithms.BaseOOD import BaseOODAlg
from GTTA.utils.config_reader import Conf
from GTTA.utils.register import register
from backup.tta_algorithms.tent import Tent


@register.tta_register
class SAR(Tent):
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Conf): Please refer to :ref:`configs:GTTA Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
                 config: Conf):

        super(SAR, self).__init__(task, model, loader, ood_algorithm, config)
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.optimizer: torch.optim.Optimizer = None
        self.episodic = config.tta.episodic
        self.steps = config.tta.SAR.steps
        num_classes = 2 if config.dataset.num_classes == 1 else config.dataset.num_classes
        self.margin_e0, self.reset_constant_em, self.ema = 0.4 * math.log(num_classes), config.tta.SAR.reset_constant_em, None
        self.config: Conf = config

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if 'convs.2.nn' in nm or 'norms.2' in nm:
                continue
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def __call__(self):
        if self.task in ['train', 'test']:
            self.load_task()
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

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, data):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        # forward
        outputs = self.model(data=data, edge_weight=None)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(
            zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = self.softmax_entropy(self.model(data=data, edge_weight=None))
        entropys2 = entropys2[filter_ids_1]  # second time forward
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(
            entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = self.update_ema(loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if self.ema is not None:
            if self.ema < self.reset_constant_em:
                print(f"ema < {self.reset_constant_em}, now reset the model")
                reset_flag = True

        return outputs, reset_flag

    def adapt_forward(self, data):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, reset_flag = self.forward_and_adapt(data)
            if reset_flag:
                self.reset()

        return outputs

    def update_ema(self, new_data):
        if self.ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * self.ema + (1 - 0.9) * new_data

    def reset(self):
        super(SAR, self).reset()
        self.ema = None


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

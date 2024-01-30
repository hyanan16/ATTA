# The implementation of ActiveTTA

Active Test-Time Adaptation: Theoretical Analyses and An Algorithm [**ICLR 2024 Paper**](https://openreview.net/forum?id=YHUGlwTzFB)

## Introduction
![Overview](/img/ATTA.png)

To advance TTA under domain shifts, we propose the novel problem setting of active test-time adaptation (ATTA) that integrates active learning within the fully TTA setting. We provide a learning theory analysis, demonstrating that incorporating limited labeled test instances enhances overall performances across test domains with a theoretical guarantee. We also present a sample entropy balancing for implementing ATTA while avoiding catastrophic forgetting (CF). We introduce a simple yet effective ATTA algorithm, known as SimATTA, using real-time sample selection techniques. Extensive experimental results confirm consistency with our theoretical analyses and show that the proposed ATTA method yields substantial performance improvements over TTA methods while maintaining efficiency and shares similar effectiveness to the more demanding active domain adaptation (ADA) methods.

## Environment setup
`environment_PyTorch21_locked.yml` is for PyTorch 2.1 environment.
The original environment is provided in `environment_PyTorch110_locked.yml`. To run the code in PyTorch 1.10,
please remove all `@torch.compile` decorators.

## Run ActiveTTA

```shell
python -m ATTA.kernel.alg_main --task train --config_p
ath TTA_configs/PACS/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.el 1e-4 --at
ta.SimATTA.nc_increase 1 --gpu_idx 0 --atta.SimATTA.LE 0 --exp_roun
d 1
python -m ATTA.kernel.alg_main --task train --config_p
ath TTA_configs/VLCS/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.el 1e-3 --at
ta.SimATTA.nc_increase 1 --gpu_idx 0 --atta.SimATTA.LE 0 --exp_roun
d 1
```

For GPU K-Means, add `--atta.gpu_clustering` to the above commands, but it may lead to slightly different results.

Pre-trained model checkpoints for PACS and VLCS are provided.
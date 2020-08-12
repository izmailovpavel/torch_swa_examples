# Stochastic Weight Averaging (SWA)
Stochastic Weight Averaging (SWA) is now natively supported in PyTorch 1.6! This repository contains examples of using the implementation of the SWA training method for DNNs in [`torch.optim.swa_utils`](https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging). The code in this repo is adapted from the original `PyTorch` [implementation](https://github.com/timgaripov/swa). Please see the new PyTorch blog post for more details about SWA and the `torch.optim` implementation. SWA was proposed in the paper

[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) (UAI 2018)

by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson. 

# Introduction

SWA is a simple DNN training method that can be used as a drop-in replacement for SGD with improved generalization, faster convergence, and essentially no overhead. The key idea of SWA is to average multiple samples produced by SGD with a modified learning rate schedule. We use a constant or cyclical learning rate schedule that causes SGD to _explore_ the set of points in the weight space corresponding to high-performing networks. We observe that SWA converges more quickly than SGD, and to wider optima that provide higher test accuracy. 

In this repo we implement the constant learning rate schedule that we found to be most practical on CIFAR datasets.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/37633888-89fdc05a-2bca-11e8-88aa-dd3661a44c3f.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/37633885-89d809a0-2bca-11e8-8d57-3bd78734cea3.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/37633887-89e93784-2bca-11e8-9d71-a385ea72ff7c.png" width=250>
</p>

Please cite our work if you find this approach useful in your research:
```latex
@article{izmailov2018averaging,
  title={Averaging Weights Leads to Wider Optima and Better Generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:1803.05407},
  year={2018}
}
```


# Dependencies
* [PyTorch 1.6 or later](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

The code in this repository implements both SWA and conventional SGD training, with examples on the CIFAR-10 and CIFAR-100 datasets.

To run SWA use the following command:

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --swa \
                 --swa_start=<SWA_START> \
                 --swa_lr=<SWA_LR>
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash; SWA learning rate (default: 0.05)


To run conventional SGD training use the following command:
```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> 
```

## Examples

To reproduce the results from the paper run (we use same parameters for both CIFAR-10 and CIFAR-100 except for PreResNet):
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 # SGD
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets

#PreResNet
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=300  --lr_init=0.1 --wd=3e-4 # SGD
#CIFAR100
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=300 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 # SWA 1.5 Budgets
#CIFAR10
python3 train.py --dir=<DIR> --dataset=CIFAR10 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=300 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets

#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=WideResNet28x10 --epochs=300 --lr_init=0.1 --wd=5e-4 # SGD
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=WideResNet28x10 --epochs=300 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 # SWA 1.5 Budgets
```

# Results

## CIFAR-100

Test accuracy (%) of SGD and SWA on CIFAR-100 for different training budgets. For each model the _Budget_ is defined as the number of epochs required to train the model with the conventional SGD procedure.

|                  | VGG-16     | ResNet-164 | WideResNet-28x10 |
|------------------|------------|------------|------------------|
| Regular Training | 72.8 ± 0.3 | 78.4 ± 0.3 | 81.0 ± 0.3       |
| SWA              | 74.4 ± 0.3 | 79.8 ± 0.4 | 82.5 ± 0.2       |

Below we show the convergence plot for SWA and SGD with PreResNet164 on CIFAR-100 and the corresponding learning rates. The dashed line illustrates the accuracy of individual models averaged by SWA.

<p align="center">
<img src="https://user-images.githubusercontent.com/14368801/37633527-226bb2d6-2bc9-11e8-9be6-097c0dfe64ab.png" width=500>
</p>
 
# References
 
Provided model implementations were adapted from
 * VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
 * PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 * WideResNet: [github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)
 
Other repos and implementations:
 * Original implementation: https://github.com/timgaripov/swa
 * PyTorch Contrib implementation: https://github.com/pytorch/contrib
 * Tensorflow implementation: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SWA
 * Chainer implementation: https://github.com/chainer/models/tree/master/swa

# Semi-Supervised Classification on CIFAR

An implementation of Semi-supervised Image Classification on the CIFAR10 and CIFAR100 datasets to show the scalability of deep neural networks in the absence of large datasets of labeled data. Deep Neural Networks that are able to classify on sparsely labeled datasets would significantly reduce the amount of manpower spent in collecting vast amounts of data and labeling them for training. The concept behind Semi-supervised Leanring is to develop strategies to simultaneously learn from labeled and unlabeled data.

This project implements 3 common techniques for SSL:
1. Pseudo-label - Dong-Hyun Lee et al.
2. Virtual adversarial training - Takeru Miyato et al.
3. A modification of Fixmatch - Kihyuk Sohn et al.

Each technique is implemented on CIFAR10 and CIFAR100 datasets with different proportions of labeled samples. For CIFAR10 which contains 5000 training images and 1000 test images, we have implemented two models. The first model contains 4000 labeled samples and 1000 unlabeled samples. The second model contains just 250 labeled samples and 4750 unlabeled samples. The difference in model performance on sparsely labeled data vs mostly labeled data will allow us to guage the extent of a deep neural network in learning from unlabeled data. Similarly, for CIFAR100 which contains 50000 training images and 10000 test images, we train models based on 2500 and 10000 labeled samples.

The model is implemented using a Wide Residual Network either using a 28-2 or 16-8 architecture using different threshold values during training to come up with the most efficient threshold values to learn from sparsely labeled data.

Sources:
1. Wide ResNets - https://github.com/szagoruyko/wide-residual-networks
2. ScienceDirect - https://www.sciencedirect.com/science/article/pii/S2405959519300694
3. Example code - https://github.com/karpathy/minGPT
4. PseudoLabel - https://github.com/iBelieveCJM/pseudo_label-pytorch
5. ResNets - https://github.com/kuangliu/pytorch-cifar/tree/master/models
6. FixMatch - https://github.com/google-research/fixmatch/blob/master/fixmatch.py
                https://github.com/LeeDoYup/FixMatch-pytorch
6. RandAugment - https://github.com/ildoonet/pytorch-randaugment

Add instructions to reproduce your reported results.

Locally, we tried smaller runs of our model to tune hyperparameters. The following were some observations that we made

1. Wide ResNet Architecture
    - We alternated between the WRN-28-2 (default) and the 16-8 architecture
    - It was noticed that the 16-8 architecture converged faster on training data than the 28-2
    - However, performance on test data while better was not as significant
    - The test loss and test accuracy seemed to stagnate after the first 15 epochs in CIFAR-10 (4000 unlabeled samples)
    - A similar stagnation was observed for all threshold values as well as the 250 unlabeled sample variation of the Pseudo-label

2. Learning Rate
    - We saw that a learning rate of 0.01 improved test accuracy more than the default 0.1
    - However, after 15 epochs, there was no visible improvement on test accuracy nor was there a decrease on test loss
    - This could be a case of vanishing gradients
 
3. Scheduler
    - We eventually decided on an initial learning rate of 0.1 used alongside a step LR scheduler of 0.2 with a Plateau Scheduler with a patience of 7 epochs
    - This led to a massive jump in test accuracy in the case of CIFAR-10 (4000) and CIFAR-100 (10000)
    - A similar jump was not noticed for CIFAR-10(250) and CIFAR-100 (2500)

4. Optimizer
    - In order to arrive at convergence faster, we used SGD with momentum as the optimizer
    - We stuck to the default momentum of 0.9
    - We reached very high rates of training accuracy (> 90%) within the first 5 epochs for CIFAR-10 and within 25 epochs for CIFAR-100
    - Perhaps a different optimizer like Adam or Adagrad would help avoid the vanishing gradient problem

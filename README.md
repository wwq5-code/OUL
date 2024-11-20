# OUbL

# Oblivious Unlearning by Learning: Machine Unlearning Without Exppsing Erased Data
## Overview
This repository is the official implementation of OUL, and the corresponding paper is under review.


## Prerequisites

```
python = 3.10.10
torch==2.0.0
torchvision==0.15.1
matplotlib==3.7.1
numpy==1.23.5
```

We also show the requirements packages in requirements.txt


## Artifact Evaluation

Here, we demonstrate the overall evaluations, which are also the main achievement claimed in the paper. We will explain the results and demonstrate how to achieve these results using the script and corresponding parameters.

Evaluated on NVIDIA Quadro RTX 6000 GPUs,

### TABLE I: General Evaluation Results on MNIST, CIFAR10 and CelebA:

On MNIST, USR = 1%

| On MNIST                  | OUL         | BFU      |   SISA   |  VBU  |
| --------                  | --------    | -------- | -------- | -------- |  
| Model Utility (Acc.)      | 98.52%      | 98.70%   |  98.53%  | 78.69%   | 
| Data Removal (Bac. Acc.)  | 9.67%       | 9.16%    | 9.67%    | 0.00%    |  
| Forgeability              | 0.942       | -        | -        | -        | 
| Reconstruction Similarity | 0.818       | 0.874    | 1        | 1        |  
| Running time (s)          | 3.920       | 16.03    | 11.70    | 0.63     |  

In this table, we can achieve these metric values by running corresponding python files.

1. To run the OUL on MNIST, we can run
```
python /OUL/OUL_experiment/On_MNIST/MNIST_OUL_R_Restart.py
```

2. To run the OUL on CIFAR10, we can run
```
python /OUL/OUL_experiment/On_CIFAR10/CIFAR10_OUL_R_Restart.py
```

3. To run the OUL on CelebA, we can run
```
python /OUL/OUL_experiment/On_CelebA/CelebA_OUL_R_Restart.py
```

Note that, to sucessfully run the program on CelebA, we need first prepare the CelebA dataset, which can be downloaded from: 
(https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
 




 
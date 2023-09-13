Covariate-informed Representation Learning to Prevent Posterior Collapse of iVAE
=====================================

This repository provides python implementation for covariate-informed identifiable variational autoencoders (CI-iVAEs) accepted for AISTATS 2023 [[paper]](https://proceedings.mlr.press/v206/kim23c/kim23c.pdf). The code is written in python with dependency in pytorch.

![image_dataset_t_sne](https://user-images.githubusercontent.com/19345323/219812373-6ab2061f-0a97-4787-99a6-d101409e9434.png)

![image_dataset_generation_results](https://user-images.githubusercontent.com/19345323/219812176-936f4e54-c379-46d6-b024-3b06f780a287.jpg)

## 1. Tutorial
### 1.1. Example code
Our main functions include ``model``, ``fit``, and ``extract_feature``. We provide an example code to use these main functions. Details on configurations are provided in ``ci_ivae_main.py``.
```python
import datetime, os, torch
import numpy as np
from sklearn.model_selection import train_test_split

import ci_ivae_main as CI_iVAE

n_train, n_test = 4000, 1000
dim_x, dim_u = 100, 5

x_train = torch.tensor(np.random.uniform(0.0, 1.0, (n_train, dim_x)), dtype=torch.float32)
u_train = torch.tensor(np.random.uniform(0.0, 1.0, (n_train, dim_u)), dtype=torch.float32)
x_test = torch.tensor(np.random.uniform(0.0, 1.0, (n_test, dim_x)), dtype=torch.float32)
u_test = torch.tensor(np.random.uniform(0.0, 1.0, (n_test, dim_u)), dtype=torch.float32)

x_train, x_val, u_train, u_val = train_test_split(x_train, u_train, test_size=(1/6))

# make result folder
now = datetime.datetime.now()
result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
os.makedirs(result_path, exist_ok=True)
print('result_path: ', result_path)

# build CI-iVAE networks
dim_x, dim_u = np.shape(x_train)[1], np.shape(u_train)[1]
ci_ivae = CI_iVAE.model(dim_x=dim_x, dim_u=dim_u)

# train CI-iVAE networks. Results will be saved at the result_path
CI_iVAE.fit(model=ci_ivae, x_train=x_train, u_train=u_train,
            x_val=x_val, u_val=u_val, num_epoch=5, result_path=result_path)

# extract features with trained CI-iVAE networks
z_train = CI_iVAE.extract_feature(result_path=result_path, x=x_train)
z_test = CI_iVAE.extract_feature(result_path=result_path, x=x_test)
z_train = z_train.detach().cpu().numpy()
z_test = z_test.detach().cpu().numpy()
```
We also provide working examples for MNIST and FashionMNIST at ``example code_MNIST.ipynb`` and ``example code_FashionMNIST.ipynb``, respectively.

### 1.2. Experimental results
For implementations to get experimental results in our experiments, please check the ``experiments`` folder.

## 2. Package dependencies
- argparse=1.1
- matplotlib=3.5.2
- numpy=1.21.4
- pandas=1.4.1
- seaborn=0.11.1
- sklearn=1.0.2
- statsmodels=0.14.0.dev0+350.g408eae829
- tensorboardX=2.4
- torch=1.10.1+cu111
- tqdm=4.40.0
- yaml=5.3.1

The code worked well with RTX 3090 and Cuda compilation tools, release 11.6, V11.6.124.

## 3. License
This project is licensed under the terms of the MIT License. This means you can freely use, modify, and distribute the code, as long as you provide attribution to the original authors and source.

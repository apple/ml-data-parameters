# Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum
This repository accompanies the research paper, [Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum](https://arxiv.org)
(accepted at NeurIPS 2019). 

## Citation
```
@article{saxena2019data,
  title={Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum},
  author={Saxena, Shreyas and Tuzel, Oncel and DeCoste, Dennis},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```
## Data Parameters
In the paper cited above, we have introduced a new family of parameters termed "data parameters".
Specifically, we equip each class and training data point with a learnable parameter (data parameters), which governs their importance during different stages of training. 
Along with the model parameters, the data parameters are also learnt with gradient descent, thereby yielding a curriculum which evolves during the course of training.
More importantly, post training, during inference, data parameters are not used, and hence do not alter the model's complexity or run-time at inference. 


# Pytorch Implementation of Conditional GAN 
### Usage
```bash
$ python3 main.py --dataset 'mnist' --epoch 50
```
NOTE: on Colab Notebook use following command:
```python
!git clone link-to-repo
%run main.py --dataset 'mnist' --epoch 50
```
### References

# Summary 

## Introduction

Generative adversarial nets were recently introduced as a novel way to train a generative model.
They consists of two ‘adversarial’ models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training
data rather than G. Both G and D could be a non-linear mapping function, such as a multi-layer perceptron.

## Minimax

Minimax is a decision rule for minimizing the possible loss for a worst case (maximum loss) scenario. 
The maximizer tries to get the highest score possible while the minimizer tries to do the opposite and get the lowest score possible.

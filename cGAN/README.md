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

Under the game theory perspective, GAN can be viewed as a game of two players: the discriminator D and the generator G. 
The discriminator tries to discriminate the generated (or fake) data and the real data, while the generator attempts to make the discriminator
confusing by gradually generating the fake data that break into the real data. 

## Conditional GAN

Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information y. We can perform the conditioning by feeding y into the both the discriminator and generator as additional input layer. 
In CGAN (Conditional GAN), labels act as an extension to the latent space z to generate and discriminate images better. Its structure can be gives as

![cGAN]
(https://golden-storage-production.s3.amazonaws.com/topic_images/23a36a66d85947c7a0fe4a2ced52914e.png)

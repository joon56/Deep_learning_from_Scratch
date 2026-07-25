# Deep Learning from Scratch

Working through *Deep Learning from Scratch* (Saito Goki, 밑바닥부터 시작하는 딥러닝)
chapter by chapter — building each component with NumPy alone, no frameworks, and
writing up every step as I went.

Started January 2023.

## Why

Reading about backpropagation and implementing it are different things. The point
of this repository is the second one: every layer, gradient and update rule here
is written out rather than called, so that the abstractions later used in PyTorch
have something concrete underneath them.

## Notebooks

| Notebook | What is implemented |
| --- | --- |
| `basicofpython.ipynb` | NumPy and Matplotlib groundwork |
| `perceptron.ipynb` | Perceptron; AND/OR/NAND, and why XOR needs a second layer |
| `nervenetwork1.ipynb` | Activation functions — step, sigmoid, ReLU |
| `nervenetwork2.ipynb` | Forward propagation as matrix multiplication |
| `nervenetwork3.ipynb` | Three-layer network, softmax, output design |
| `nervenetwork4.ipynb` | MNIST inference with pretrained weights, batching |
| `nervenetwork learning(1).ipynb` | Loss functions — MSE and cross-entropy |
| `nervenetwork_learning(2).ipynb` | Mini-batch learning |
| `nervenetwork_learning(3).ipynb` | Numerical differentiation and gradient descent |
| `backpropagation.ipynb` | Computational graphs; `Relu`, `Sigmoid`, `Affine`, `SoftmaxWithLoss` as layer classes; `TwoLayerNet`; gradient check against numerical differentiation |
| `conv_net(1).ipynb` | `im2col`, and `Convolution` / `Pooling` forward passes |
| `conv_net(2).ipynb` | `SimpleConvNet` — Conv → ReLU → Pool → Affine → ReLU → Affine → Softmax |

Dataset throughout is MNIST. The notebooks import the book's companion modules
(`dataset.mnist`, `common.layers`, `common.gradient`, `common.trainer`), so clone
[the book's official repository](https://github.com/WegraLee/deep-learning-from-scratch)
alongside these and run them from there — the imports will not resolve otherwise.

The gradient check in `backpropagation.ipynb` is the one output worth looking at:
analytic and numerical gradients agree to roughly 1e-6–1e-9, which is what tells
you the hand-written backward passes are correct.

## Write-ups

Each notebook has a companion post, in Korean, on my blog.

- [Python basics](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B8%B0%EC%B4%88)
- [Perceptron](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)
- Neural networks —
  [1](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EA%B3%BC-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98) ·
  [2](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-2) ·
  [3](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-3) ·
  [4 (MNIST)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-4-MNIST-%EC%86%90%EA%B8%80%EC%94%A8-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EC%8B%9D)
- Training —
  [1](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5-1) ·
  [2 (mini-batch)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5-2-%EB%AF%B8%EB%8B%88-%EB%B0%B0%EC%B9%98-%ED%95%99%EC%8A%B5) ·
  [3 (gradients)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5-3-%EC%88%98%EC%B9%98-%EB%AF%B8%EB%B6%84%EA%B3%BC-%EA%B8%B0%EC%9A%B8%EA%B8%B0) ·
  [4 (algorithm)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5-4-%ED%95%99%EC%8A%B5-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B5%AC%ED%98%84)
- Backpropagation —
  [1 (chain rule)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C%EB%B2%95-1-%EC%97%AD%EC%A0%84%ED%8C%8C%EC%99%80-%EA%B3%84%EC%82%B0%EB%B2%95%EC%B9%99) ·
  [2 (layers)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C%EB%B2%95-2-%EA%B3%84%EC%B8%B5) ·
  [3 (implementation)](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C%EB%B2%95-3-%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C%EB%B2%95-%EA%B5%AC%ED%98%84)
- Convolutional networks —
  [1](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-1) ·
  [2](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-2) ·
  [3](https://helloparzival.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-3)

Two of these posts — training 4 and ConvNet 3 — have no matching notebook in this
repository.

## Where this went next

The MRI slice-selection work in
[AD_Classifying_AI_Model](https://github.com/joon56/AD_Classifying_AI_Model),
and a 2023 paper on Alzheimer's classification from brain MRI.

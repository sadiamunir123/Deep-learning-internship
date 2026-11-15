# Deep Learning Phase 2: Convolutional Neural Networks

Phase 2 of my deep learning internship work. In one week I went from the basic idea of convolution to building CNNs in PyTorch, improving them with batch normalization, dropout and data augmentation, understanding ResNet, and finishing with transfer learning.

**Datasets:** Fashion-MNIST and CIFAR-10 (both download automatically with `torchvision`)
**Framework:** PyTorch

## Notebooks

| Day | Notebook | Topic |
|---|---|---|
| 1 | [day1_convolution_from_scratch.ipynb](day1_convolution_from_scratch.ipynb) | Convolution, padding, stride and pooling written by hand in NumPy; edge and blur filters; output-size formula; parameter count of conv vs fully connected layers |
| 2 | [day2_cnn_vs_mlp_fashion_mnist.ipynb](day2_cnn_vs_mlp_fashion_mnist.ipynb) | First CNN in PyTorch on Fashion-MNIST; comparison with an MLP baseline |
| 3 | [day3_cnn_batchnorm_dropout_fashion_mnist.ipynb](day3_cnn_batchnorm_dropout_fashion_mnist.ipynb) | Main project: CNN with batch norm and dropout, test evaluation, confusion matrix, misclassified images, learned filters and feature maps |
| 4 | [day4_cifar10_data_augmentation.ipynb](day4_cifar10_data_augmentation.ipynb) | Colour images (CIFAR-10), overfitting, and data augmentation (flip, crop) with and without |
| 5 | [day5_resnet_architectures.ipynb](day5_resnet_architectures.ipynb) | LeNet, VGG and ResNet; residual block and mini-ResNet vs a plain CNN; ResNet18 structure |
| 6 | [day6_transfer_learning_resnet18.ipynb](day6_transfer_learning_resnet18.ipynb) | Transfer learning with pretrained ResNet18: feature extraction, fine-tuning, and comparison with training from scratch |

## What I Learned

- How an image is stored as numbers and how a convolution slides a filter over it to create a feature map
- Padding, stride, pooling, and the output-size formula `floor((n + 2p - f) / s) + 1`
- Why CNNs use far fewer parameters than MLPs on images (local patterns and shared weights)
- Building, training and evaluating CNNs in PyTorch, including GPU training and tracking tensor shapes
- Using a proper train / validation / test split and evaluating on the test set only once
- Batch normalization, dropout and data augmentation, and what each one does
- Reading a confusion matrix and visualising what a CNN has learned
- Residual (skip) connections and why they make deep networks trainable
- Transfer learning: feature extraction vs fine-tuning

## Results

Fill in from the outputs of each notebook after running it.

| Experiment | Dataset | Model | Test accuracy | Parameters |
|---|---|---|---|---|
| MLP baseline | Fashion-MNIST | MLP | | |
| Small CNN | Fashion-MNIST | CNN | | |
| CNN + batch norm + dropout | Fashion-MNIST | CNN | | |
| No augmentation | CIFAR-10 | CNN | | |
| With augmentation | CIFAR-10 | CNN | | |
| Plain CNN | CIFAR-10 | CNN | | |
| Mini-ResNet | CIFAR-10 | ResNet | | |
| Feature extraction | CIFAR-10 (5,000 images) | ResNet18 (pretrained) | | |
| Fine-tuning | CIFAR-10 (5,000 images) | ResNet18 (pretrained) | | |
| From scratch | CIFAR-10 (5,000 images) | CNN | | |

## Key Concepts Explained

- **Convolution:** a small filter (kernel) slides over the image; at each position the values are multiplied and summed to give one number in the feature map. The filter values are learned during training.
- **Batch normalization:** normalises the outputs of a layer for each batch, which makes training faster and more stable.
- **Dropout:** randomly switches off neurons during training, so the model cannot rely on any single one and overfits less. It is turned off at evaluation time (`model.eval()`).
- **Data augmentation:** random flips and crops create new variants of each training image every epoch. It is applied only to the training set.
- **Residual connection:** a block outputs `F(x) + x`, so the layers only learn the change. The skip path lets gradients flow back easily, which allows very deep networks.
- **Transfer learning:** reuse a network pretrained on ImageNet. Feature extraction trains only the new last layer; fine-tuning also trains some pretrained layers with a small learning rate.

## Key Takeaways

- Understanding the shape of the tensor after every layer prevents most CNN bugs.
- Compare training and validation curves to spot overfitting, and keep the test set for one final evaluation.
- Augmentation, batch norm and dropout are simple ways to improve generalisation.
- With little data, a pretrained model usually beats training from scratch.

## How to Run

**Recommended: Google Colab with a GPU**
1. Upload a notebook to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type > T4 GPU**.
3. Run all cells. Datasets download automatically, so nothing needs to be uploaded.

**Locally (VS Code or Jupyter)**
```
py -m pip install -r requirements.txt
```
Then open a notebook and select the Python environment where the packages are installed as the kernel. A CPU works, but CIFAR-10 and transfer learning are much slower, so reduce the number of epochs (and use `IMG_SIZE = 128` in Day 6).

## Requirements

See [requirements.txt](requirements.txt): `torch`, `torchvision`, `numpy`, `matplotlib`, `jupyter`.

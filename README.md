
# Gomma Lab

A Python toolkit for practising Data Analytics laboratory experiments.

## Experiments

1. Gradient Descent
2. Linear and Binary Search
3. K-Means Clustering
4. SVM Classification
5. Image Classification using ResNet50
6. LSTM Time Series Prediction

## Installation

```bash
pip install gomma-lab
```

## Usage

```python
from gomma_lab import searching

print(searching.linear_search([10, 20, 30], 20))
```

### Gradient Descent

```python
from gomma_lab.gradient import gradient_descent

gradient_descent()
```

### K-Means

```python
from gomma_lab.clustering import kmeans_clustering

kmeans_clustering()
```

### SVM Classification

```python
from gomma_lab.classification import svm_classification

svm_classification()
```

### Image Classification

```python
from gomma_lab.cnn import classify_image

classify_image("image.jpg")
```

### LSTM

```python
from gomma_lab.lstm import lstm_prediction

lstm_prediction()
```

## Requirements

Python 3.9 or later.

CNN and LSTM require PyTorch.
CNN also downloads pretrained ResNet50 weights on first use.

## License

See LICENSE.

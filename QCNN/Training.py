# Implementation of Quantum circuit training procedure
from typing import Optional

import autograd.numpy as anp
import Hierarchical_circuit
import pennylane as qml
import QCNN_circuit
from pennylane import numpy as np


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def multi_class_cross_entropy(labels, predictions, num_classes: Optional[int] = 10):
    """
    Compute the cross-entropy loss between ground truth labels and predicted probabilities.

    Args:
    - labels (array): Ground truth labels, shape (num_samples,)
    - predictions (array): Predicted probabilities for each class, shape (num_samples, num_classes)

    Returns:
    - loss (float): Cross-entropy loss
    """
    num_samples = len(labels)
    loss = 0
    for i in range(num_samples):
        label = labels[i]
        prediction = predictions[i]
        # FIXME: as the probabilites from qml are not in between 0 and 1, we need to normalize them apply softmax function
        softmax_probabilites = softmax(prediction)
        # testing the predicted class label
        # predicted_class_label = np.argmax(softmax_probabilites)
        c_entropy = -np.log(softmax_probabilites[label])
        loss += c_entropy
    return loss / num_samples


def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == "QCNN":
        # we need predictions for 10 classes only
        predictions = [
            QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn)[:10]
            for x in X
        ]
    elif circuit == "Hierarchical":
        predictions = [
            Hierarchical_circuit.Hierarchical_classifier(
                x, params, U, U_params, embedding_type, cost_fn=cost_fn
            )
            for x in X
        ]

    if cost_fn == "mse":
        loss = square_loss(Y, predictions)
    elif cost_fn == "cross_entropy":
        if U == "U_SU4_mod":
            loss = multi_class_cross_entropy(Y, predictions)
        else:
            loss = cross_entropy(Y, predictions)
    return loss


# Circuit training parameters
steps = 200
learning_rate = 0.01
batch_size = 25


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == "QCNN":
        if U == "U_SU4_no_pooling" or U == "U_SU4_1D" or U == "U_9_1D":
            total_params = U_params * 3
        else:
            total_params = U_params * 3 + 2 * 3
    elif circuit == "Hierarchical":
        total_params = U_params * 7
    # randomly initialize the parameters
    params = np.random.randn(total_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []
    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
            params,
        )
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    return loss_history, params

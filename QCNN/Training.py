# Implementation of Quantum circuit training procedure
from typing import List, Optional

import autograd.numpy as anp
import Hierarchical_circuit
import pennylane as qml
import QCNN_circuit
import torch
from pennylane import numpy as np

loss_function = torch.nn.CrossEntropyLoss()


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


def multi_class_cross_entropy(labels: List, predictions: List, num_classes: Optional[int] = 10):
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
        c_entropy = -anp.log(softmax_probabilites[label])
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


def get_predicted_labels_QCNN(params, X, Y, U, U_params, embedding_type, cost_fn):
    # we need predictions for 10 classes only
    predictions = [
        QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn)[:10] for x in X
    ]
    softmax_predictions = [softmax(p) for p in predictions]
    # get predicted labels
    predicted_labels = np.argmax(softmax_predictions, axis=1)
    return predicted_labels


# Circuit training parameters
epochs = 1
learning_rate = 0.001
batch_size = 25


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == "QCNN":
        if U == "U_SU4_no_pooling" or U == "U_SU4_1D" or U == "U_9_1D":
            total_params = U_params * 3
        else:
            total_params = U_params * 3 + 2 * 3
    elif circuit == "Hierarchical":
        total_params = U_params * 7
    steps_per_epoch = len(X_train) // batch_size
    # randomly initialize the parameters
    params = np.random.randn(total_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    # opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []
    acc_history = []
    for it in range(epochs):
        total_samples = 0
        total_loss = 0
        correct_count = 0
        # shuffle the data for each epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        for step in range(steps_per_epoch):
            # create mini-batch
            X_batch = X_train_shuffled[step * batch_size : (step + 1) * batch_size]
            Y_batch = Y_train_shuffled[step * batch_size : (step + 1) * batch_size]
            prev_params = params
            params, cost_new = opt.step_and_cost(
                lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
                params,
            )
            predicted_labels = get_predicted_labels_QCNN(
                prev_params, X_batch, Y_batch, U, U_params, embedding_type, cost_fn
            )
            # print(f"Predicted labels: {predicted_labels}")
            # print(f"True labels: {Y_batch}")
            correct_count += np.sum(predicted_labels == Y_batch)
            total_samples += len(Y_batch)
            total_loss += cost_new * len(Y_batch)
        # average accuracy for the epoch
        accuracy = correct_count / total_samples
        # average loss for the epoch
        average_loss = total_loss / total_samples
        loss_history.append(average_loss)
        if isinstance(accuracy, qml.numpy.tensor):
            accuracy = accuracy.item()
        acc_history.append(accuracy)
        # log details
        print(f"Epoch {it + 1}/{epochs}, Average Loss: {average_loss:.4f}")
        print(f"Epoch {it + 1}/{epochs}, Accuracy: {accuracy:.4f}")
    return loss_history, acc_history, params

# This generates the results of the bechmarking code

import Benchmarking
import plotly.graph_objs as go

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
Encodings: ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact', 'pca32-1', 'autoencoder32-1',
            'pca16-1', 'autoencoder16-1', 'pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1']
dataset: 'mnist' or 'fashion_mnist'
circuit: 'QCNN' or 'Hierarchical'
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

# Unitaries = ["U_SU4", "U_SU4_1D", "U_SU4_no_pooling", "U_9_1D"]
Unitaries = ["U_SU4_mod"]
# U_num_params = [15, 15, 15, 2]
U_num_params = [15]
Encodings = ["resize256"]
# dataset = "fashion_mnist"
dataset = "mnist"
# classes = "0_1"
classes = "all"
binary = False
multi_class = True
cost_fn = "cross_entropy"
plot_circuit = True

Benchmarking.Benchmarking(
    dataset,
    classes,
    Unitaries,
    U_num_params,
    Encodings,
    circuit="QCNN",
    multi_class=multi_class,
    cost_fn=cost_fn,
    binary=binary,
    plot_circuit=plot_circuit,
)
# Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='Hierarchical', cost_fn=cost_fn, binary=binary)


def plot_loss_history_QCNN_mod():
    # Data input
    data = [
        2.302457621446221,
        2.3012750662921864,
        2.300081672034619,
        2.2989030812757933,
        2.2977365740182307,
        2.296672793211272,
        2.295752130598106,
        2.2949501079912316,
        2.2942215639943173,
        2.2936265210536897,
        2.2930708621399067,
        2.292582728828179,
        2.2921405792559497,
        2.291723862428457,
        2.291347785723987,
        2.291000066832412,
        2.290672165051221,
        2.2903659564412115,
        2.2900683567280833,
        2.289784306600893,
        2.2894969423758913,
        2.289227453900381,
        2.288952607164454,
        2.2886770996907724,
        2.288408828039322,
        2.2881449563579013,
        2.287860597561603,
        2.287585352049889,
        2.2873073949308935,
        2.287061939274921,
    ]

    # Create the plot
    fig = go.Figure()

    # Add the data trace
    fig.add_trace(go.Scatter(y=data, mode="lines", name="Training Loss"))

    # Update layout
    fig.update_layout(
        title="Training Loss Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Legend",
    )
    # Save the plot
    fig.write_image("training_loss_plot.png")


def plot_train_acc_history_QCNN_mod():
    # Data input
    data = [
        0.098,
        0.119,
        0.122,
        0.117,
        0.118,
        0.125,
        0.132,
        0.137,
        0.139,
        0.141,
        0.145,
        0.143,
        0.142,
        0.144,
        0.145,
        0.147,
        0.147,
        0.152,
        0.153,
        0.155,
        0.154,
        0.159,
        0.158,
        0.156,
        0.158,
        0.161,
        0.163,
        0.165,
        0.166,
        0.168,
    ]

    # Create the plot
    fig = go.Figure()

    # Add the data trace
    fig.add_trace(go.Scatter(y=data, mode="lines", name="Training Acc"))

    # Update layout
    fig.update_layout(
        title="Training Acc Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Acc",
        legend_title="Legend",
    )
    # Save the plot
    fig.write_image("training_acc_plot.png")

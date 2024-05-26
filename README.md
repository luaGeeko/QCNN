# QOSF_project

This is an implementation of Quantum convolutional neural network for classical data classification (https://arxiv.org/abs/2108.00661). It uses Pennylane software (https://pennylane.ai) for classifying MNIST and Fashion MNIST datasets.

### 1. Data Preparation
**"data.py"**: loads dataset (MNIST or Fashion MNIST) after classical preprocessing. 
"feature_reduction" argument contains information of preprocessing method and     dimension of preprocessed data.

### 2. QCNN Circuit
**"QCNN_circuit.py"**: QCNN function implements Quantum Convolutional Neural Network.
"QCNN_structure" is standard structure with iterations of convolution and pooling layers.
"QCNN_structure_without_pooling" has only convolution layers.
"QCNN_1D_circuit" has connection between first and last qubits removed.
(**"Hierarchical.py"**: similarily for Hierarchical Quantum Classifier structure, https://www.nature.com/articles/s41534-018-0116-9)

**"unitary.py"**: contains all the unitary ansatz used for convolution and pooling layers.

**"embedding.py"**: shows how classical data is initially embedded into QCNN circuit.
There are five main embeddings: Amplitude Embedding, Angle Embedding, Compact Angle Embedding ("Angle-compact"), Hybrid Direct Embedding ("Amplitude-Hybrid"), Hybrid Angle Embedding ("Angulr-Hybrid").

**"Angular_hybrid.py"**: contains Hybrid Angle Embedding structure used in **"embedding.py"**

Hybrid Direct Embedding and Hybrid Angle Embedding have variations depending on number of qubits in a embedding block and embedding arrangements. For example, "Amplitude-Hybrid4-1" embeds 16 classical data in 4 qubits in arrangement [[0,1,2,3], [4,5,6,7]].

### 3. QCNN Training
**"Training.py"**: trains quantum circuit (QCNN or Hierarchical). By default, it uses nesterov momentum optimizer to train 200 iteartions with batch size of 25. Both MSE Loss and cross entropy loss can be used for circuit training. Number of total parameters (total_params) need to be adjusted when testing different QCNN structures. 

### 4. Benchmarking
**"Benchmarking.py"**: trains quantum circuit for given dataset, unitary ansatze, and encoding / embedding method. Saves training loss history and test data accuracy after training. Encoding_to_Embedding function converts Encoding (classical preprocessing feature reduction) to Embedding (Classical data embedding into Quantum Circuit).


Binary: "True" uses 1 and -1 labels, while "False" uses 1 and 0 labels. When using cross entropy cost function always use "False".
When using mse cost function "True" in result for paper, but "False" can also be used.

### 4. Using Docker Build
1. Install docker and docker compose according to your system requirements
2. Clone the repository in your system
3. `cd QCNN` ---> `sudo docker compose build` OR `sudo docker-compose build`, same commands but depends on docker installation
4. Once the docker builds the image `sudo docker compose run --rm app bash`
5. `python result.py` --> entry point script for training any of the QCNN model and contains initial parameters needed, default have been set.
6. To train for multi - class [WIP] model `"U_SU4_mod"`, `classes = "all"`, `multi_class = True`
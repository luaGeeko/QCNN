# This is a mean and standard deviation of 1D chain QCNN with SU4 unitary ansatz
import numpy as np

U9_resize256_MNIST = np.array(
    [
        0.858628841607565,
        0.9460992907801419,
        0.9281323877068558,
        0.798581560283688,
        0.7976359338061466,
    ]
)
U9_AE8_MNIST = np.array(
    [
        0.851063829787234,
        0.9550827423167849,
        0.9309692671394799,
        0.968321513002364,
        0.7111111111111111,
    ]
)
U9_PCA8_MNIST = np.array(
    [
        0.9862884160756501,
        0.9782505910165484,
        0.9754137115839243,
        0.9782505910165484,
        0.9777777777777777,
    ]
)
U9_PCA16_MNIST = np.array(
    [
        0.9427895981087471,
        0.9304964539007092,
        0.975886524822695,
        0.9687943262411347,
        0.7182033096926714,
    ]
)
U9_AE16_MNIST = np.array(
    [
        0.9456264775413712,
        0.9711583924349881,
        0.891725768321513,
        0.7990543735224587,
        0.9200945626477541,
    ]
)

SU4_resize256_MNIST = np.array(
    [
        0.9673758865248226,
        0.9650118203309692,
        0.966903073286052,
        0.9829787234042553,
        0.9361702127659575,
    ]
)
SU4_AE8_MNIST = np.array(
    [
        0.957919621749409,
        0.9229314420803783,
        0.9867612293144208,
        0.9735224586288416,
        0.9919621749408983,
    ]
)
SU4_PCA8_MNIST = np.array(
    [
        0.9791962174940898,
        0.983451536643026,
        0.9829787234042553,
        0.9900709219858156,
        0.9777777777777777,
    ]
)
SU4_PCA16_MNIST = np.array(
    [
        0.9801418439716312,
        0.9829787234042553,
        0.9801418439716312,
        0.983451536643026,
        0.9768321513002364,
    ]
)
SU4_AE16_MNIST = np.array(
    [
        0.9470449172576832,
        0.9640661938534278,
        0.9139479905437352,
        0.9385342789598109,
        0.9702127659574468,
    ]
)

U9_resize256_FASHION = np.array([0.7795, 0.876, 0.8555, 0.698, 0.9395])
U9_AE8_FASHION = np.array([0.9135, 0.7215, 0.852, 0.9035, 0.7855])
U9_PCA8_FASHION = np.array([0.892, 0.8435, 0.87, 0.797, 0.8405])
U9_PCA16_FASHION = np.array([0.5475, 0.7535, 0.788, 0.8305, 0.8285])
U9_AE16_FASHION = np.array([0.649, 0.8535, 0.847, 0.9165, 0.8425])

SU4_resize256_FASHION = np.array([0.925, 0.8935, 0.902, 0.9115, 0.9045])
SU4_AE8_FASHION = np.array([0.8995, 0.934, 0.8435, 0.9215, 0.932])
SU4_PCA8_FASHION = np.array([0.921, 0.916, 0.9295, 0.934, 0.9285])
SU4_PCA16_FASHION = np.array([0.9265, 0.926, 0.9335, 0.928, 0.914])
SU4_AE16_FASHION = np.array([0.934, 0.9445, 0.8915, 0.927, 0.918])


print("Result for MNIST dataset with 1D chain QCNN structure with MSELoss\n")
print("Result with U_9: \n")
print("resize256: " + str(U9_resize256_MNIST.mean()) + " +/- " + str(U9_resize256_MNIST.std()))
print("PCA8: " + str(U9_PCA8_MNIST.mean()) + " +/- " + str(U9_PCA8_MNIST.std()))
print("AE8: " + str(U9_AE8_MNIST.mean()) + " +/- " + str(U9_AE8_MNIST.std()))
print("PCA16: " + str(U9_PCA16_MNIST.mean()) + " +/- " + str(U9_PCA16_MNIST.std()))
print("AE16: " + str(U9_AE16_MNIST.mean()) + " +/- " + str(U9_AE16_MNIST.std()))
print("Result with SU4: \n")
print("resize256: " + str(SU4_resize256_MNIST.mean()) + " +/- " + str(SU4_resize256_MNIST.std()))
print("PCA8: " + str(SU4_PCA8_MNIST.mean()) + " +/- " + str(SU4_PCA8_MNIST.std()))
print("AE8: " + str(SU4_AE8_MNIST.mean()) + " +/- " + str(SU4_AE8_MNIST.std()))
print("PCA16: " + str(SU4_PCA16_MNIST.mean()) + " +/- " + str(SU4_PCA16_MNIST.std()))
print("AE16: " + str(SU4_AE16_MNIST.mean()) + " +/- " + str(SU4_AE16_MNIST.std()))

print("Result for Fashion MNIST dataset with 1D chain QCNN structure\n")
print("Result with U_9: \n")
print("resize256: " + str(U9_resize256_FASHION.mean()) + " +/- " + str(U9_resize256_FASHION.std()))
print("PCA8: " + str(U9_PCA8_FASHION.mean()) + " +/- " + str(U9_PCA8_FASHION.std()))
print("AE8: " + str(U9_AE8_FASHION.mean()) + " +/- " + str(U9_AE8_FASHION.std()))
print("PCA16: " + str(U9_PCA16_FASHION.mean()) + " +/- " + str(U9_PCA16_FASHION.std()))
print("AE16: " + str(U9_AE16_FASHION.mean()) + " +/- " + str(U9_AE16_FASHION.std()))
print("Result with SU4: \n")
print(
    "resize256: " + str(SU4_resize256_FASHION.mean()) + " +/- " + str(SU4_resize256_FASHION.std())
)
print("PCA8: " + str(SU4_PCA8_FASHION.mean()) + " +/- " + str(SU4_PCA8_FASHION.std()))
print("AE8: " + str(SU4_AE8_FASHION.mean()) + " +/- " + str(SU4_AE8_FASHION.std()))
print("PCA16: " + str(SU4_PCA16_FASHION.mean()) + " +/- " + str(SU4_PCA16_FASHION.std()))
print("AE16: " + str(SU4_AE16_FASHION.mean()) + " +/- " + str(SU4_AE16_FASHION.std()))

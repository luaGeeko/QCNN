# This is a mean and standard deviation of 1D chain QCNN with SU4 unitary ansatz
import numpy as np

U9_resize256_MNIST = np.array(
    [
        0.8969267139479905,
        0.866193853427896,
        0.9016548463356974,
        0.9437352245862884,
        0.9111111111111111,
    ]
)
U9_PCA8_MNIST = np.array(
    [
        0.9763593380614657,
        0.9825059101654846,
        0.9782505910165484,
        0.9829787234042553,
        0.9810874704491725,
    ]
)
U9_AE8_MNIST = np.array(
    [
        0.9744680851063829,
        0.9895981087470449,
        0.9952718676122931,
        0.9494089834515367,
        0.9068557919621749,
    ]
)
U9_PCA16_MNIST = np.array(
    [
        0.9763593380614657,
        0.9763593380614657,
        0.9796690307328605,
        0.9763593380614657,
        0.9763593380614657,
    ]
)
U9_AE16_MNIST = np.array(
    [
        0.8326241134751773,
        0.9219858156028369,
        0.9101654846335697,
        0.8581560283687943,
        0.7990543735224587,
    ]
)

SU4_resize256_MNIST = np.array(
    [
        0.9806146572104019,
        0.9891252955082742,
        0.9739952718676123,
        0.9881796690307328,
        0.9877068557919622,
    ]
)
SU4_PCA8_MNIST = np.array(
    [
        0.983451536643026,
        0.9796690307328605,
        0.9796690307328605,
        0.9806146572104019,
        0.984869976359338,
    ]
)
SU4_AE8_MNIST = np.array(
    [
        0.9791962174940898,
        0.9891252955082742,
        0.9531914893617022,
        0.968321513002364,
        0.9309692671394799,
    ]
)
SU4_PCA16_MNIST = np.array(
    [
        0.9787234042553191,
        0.9754137115839243,
        0.9862884160756501,
        0.9782505910165484,
        0.9810874704491725,
    ]
)
SU4_AE16_MNIST = np.array(
    [
        0.9739952718676123,
        0.983451536643026,
        0.9626477541371158,
        0.9872340425531915,
        0.9735224586288416,
    ]
)

U9_resize256_FASHION = np.array([0.913, 0.9025, 0.888, 0.9295, 0.9195])
U9_PCA8_FASHION = np.array([0.8665, 0.866, 0.856, 0.8575, 0.866])
U9_AE8_FASHION = np.array([0.9335, 0.8895, 0.883, 0.8425, 0.8695])
U9_PCA16_FASHION = np.array([0.7475, 0.8275, 0.872, 0.8745, 0.842])
U9_AE16_FASHION = np.array([0.935, 0.903, 0.9055, 0.7765, 0.8405])

SU4_resize256_FASHION = np.array([0.926, 0.8985, 0.9155, 0.929, 0.914])
SU4_PCA8_FASHION = np.array([0.923, 0.9265, 0.92, 0.9095, 0.921])
SU4_AE8_FASHION = np.array([0.899, 0.9305, 0.9125, 0.9375, 0.9455])
SU4_PCA16_FASHION = np.array([0.8835, 0.919, 0.923, 0.8925, 0.917])
SU4_AE16_FASHION = np.array([0.9395, 0.9115, 0.9345, 0.94, 0.94])


print("Result for MNIST dataset with 1D chain QCNN structure with CrossEntropyLoss")
print("Result with U_9: ")
print("resize256: " + str(U9_resize256_MNIST.mean()) + " +/- " + str(U9_resize256_MNIST.std()))
print("PCA8: " + str(U9_PCA8_MNIST.mean()) + " +/- " + str(U9_PCA8_MNIST.std()))
print("AE8: " + str(U9_AE8_MNIST.mean()) + " +/- " + str(U9_AE8_MNIST.std()))
print("PCA16: " + str(U9_PCA16_MNIST.mean()) + " +/- " + str(U9_PCA16_MNIST.std()))
print("AE16: " + str(U9_AE16_MNIST.mean()) + " +/- " + str(U9_AE16_MNIST.std()))
print("\n")
print("Result with SU4: ")
print("resize256: " + str(SU4_resize256_MNIST.mean()) + " +/- " + str(SU4_resize256_MNIST.std()))
print("PCA8: " + str(SU4_PCA8_MNIST.mean()) + " +/- " + str(SU4_PCA8_MNIST.std()))
print("AE8: " + str(SU4_AE8_MNIST.mean()) + " +/- " + str(SU4_AE8_MNIST.std()))
print("PCA16: " + str(SU4_PCA16_MNIST.mean()) + " +/- " + str(SU4_PCA16_MNIST.std()))
print("AE16: " + str(SU4_AE16_MNIST.mean()) + " +/- " + str(SU4_AE16_MNIST.std()))
print("\n")

print("Result for Fashion MNIST dataset with 1D chain QCNN structure")
print("Result with U_9: ")
print("resize256: " + str(U9_resize256_FASHION.mean()) + " +/- " + str(U9_resize256_FASHION.std()))
print("PCA8: " + str(U9_PCA8_FASHION.mean()) + " +/- " + str(U9_PCA8_FASHION.std()))
print("AE8: " + str(U9_AE8_FASHION.mean()) + " +/- " + str(U9_AE8_FASHION.std()))
print("PCA16: " + str(U9_PCA16_FASHION.mean()) + " +/- " + str(U9_PCA16_FASHION.std()))
print("AE16: " + str(U9_AE16_FASHION.mean()) + " +/- " + str(U9_AE16_FASHION.std()))
print("\n")
print("Result with SU4: ")
print(
    "resize256: " + str(SU4_resize256_FASHION.mean()) + " +/- " + str(SU4_resize256_FASHION.std())
)
print("PCA8: " + str(SU4_PCA8_FASHION.mean()) + " +/- " + str(SU4_PCA8_FASHION.std()))
print("AE8: " + str(SU4_AE8_FASHION.mean()) + " +/- " + str(SU4_AE8_FASHION.std()))
print("PCA16: " + str(SU4_PCA16_FASHION.mean()) + " +/- " + str(SU4_PCA16_FASHION.std()))
print("AE16: " + str(SU4_AE16_FASHION.mean()) + " +/- " + str(SU4_AE16_FASHION.std()))

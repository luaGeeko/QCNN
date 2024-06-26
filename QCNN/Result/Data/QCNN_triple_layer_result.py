# This is a mean and standard deviation of Triple Layer QCNN with U9 and SU4 unitary ansatz
import numpy as np

U9_resize256_MNIST = np.array(
    [
        0.924822695035461,
        0.9011820330969267,
        0.9200945626477541,
        0.9739952718676123,
        0.9716312056737588,
    ]
)
U9_AE8_MNIST = np.array(
    [
        0.9408983451536643,
        0.9801418439716312,
        0.9706855791962175,
        0.9385342789598109,
        0.9229314420803783,
    ]
)
U9_PCA8_MNIST = np.array(
    [
        0.9546099290780142,
        0.9806146572104019,
        0.9773049645390071,
        0.9574468085106383,
        0.9517730496453901,
    ]
)
U9_PCA16_MNIST = np.array(
    [
        0.984869976359338,
        0.9796690307328605,
        0.9806146572104019,
        0.9815602836879432,
        0.9569739952718677,
    ]
)
U9_AE16_MNIST = np.array(
    [
        0.9588652482269504,
        0.9092198581560283,
        0.9508274231678487,
        0.8846335697399527,
        0.9205673758865248,
    ]
)

SU4_resize256_MNIST = np.array(
    [
        0.9820330969267139,
        0.984869976359338,
        0.9801418439716312,
        0.9825059101654846,
        0.9867612293144208,
    ]
)
SU4_AE8_MNIST = np.array(
    [
        0.9749408983451536,
        0.9862884160756501,
        0.9867612293144208,
        0.9858156028368794,
        0.9872340425531915,
    ]
)
SU4_PCA8_MNIST = np.array(
    [
        0.984869976359338,
        0.9891252955082742,
        0.9810874704491725,
        0.9853427895981087,
        0.9791962174940898,
    ]
)
SU4_PCA16_MNIST = np.array(
    [
        0.9858156028368794,
        0.9801418439716312,
        0.984869976359338,
        0.984869976359338,
        0.9853427895981087,
    ]
)
SU4_AE16_MNIST = np.array(
    [
        0.9725768321513002,
        0.9333333333333333,
        0.9744680851063829,
        0.9678486997635933,
        0.992434988179669,
    ]
)

U9_resize256_FASHION = np.array([0.9225, 0.918, 0.8975, 0.9225, 0.905])
U9_AE8_FASHION = np.array([0.9105, 0.8555, 0.921, 0.942, 0.927])
U9_PCA8_FASHION = np.array([0.868, 0.8745, 0.85, 0.828, 0.8385])
U9_PCA16_FASHION = np.array([0.9115, 0.847, 0.909, 0.852, 0.8605])
U9_AE16_FASHION = np.array([0.9245, 0.894, 0.889, 0.888, 0.9065])

SU4_resize256_FASHION = np.array([0.9145, 0.8915, 0.9165, 0.8985, 0.8945])
SU4_AE8_FASHION = np.array([0.9205, 0.93, 0.941, 0.931, 0.9495])
SU4_PCA8_FASHION = np.array([0.8995, 0.882, 0.875, 0.9055, 0.881])
SU4_PCA16_FASHION = np.array([0.9155, 0.8885, 0.901, 0.879, 0.881])
SU4_AE16_FASHION = np.array([0.936, 0.933, 0.937, 0.9505, 0.95])


print("Result for MNIST dataset with Triple Layer QCNN structure\n")
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

print("Result for Fashion MNIST dataset with Triple Layer QCNN structure\n")
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

import MNISTloader
import ANNScratchMNIST

images, labels = MNISTloader.load_data(True)
training_data = [(images[i], labels[i]) for i in range(int(len(images)))]

net = ANNScratchMNIST.Network([784, 30, 10], 0)
net2 = ANNScratchMNIST.Network([784, 30, 10], 0)

net.SGD(training_data, 5, 28, 0.5)
print("\n \n")
net2.HPT_SGD(training_data, 5, 28, 0.5)

# net.HPT_SGD(training_data, 5, 28, 0.5)
# net.hebbian("BCM", training_data, 20, 0.5)

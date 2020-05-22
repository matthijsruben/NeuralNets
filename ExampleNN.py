import MNISTloader
import ANNScratchMNIST

images, labels = MNISTloader.load_data(True)
training_data = [(images[i], labels[i]) for i in range(int(len(images)))]

net = ANNScratchMNIST.Network([784, 30, 10])
net.SGD(training_data, 20, 28, 0.5)
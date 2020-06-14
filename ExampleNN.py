import MNISTloader
import ANNScratchMNIST
import matplotlib.pyplot as plt
import numpy as np

images, labels = MNISTloader.load_data(True)
training_data = [(images[i], labels[i]) for i in range(int(len(images)))]

net = ANNScratchMNIST.Network([784, 30, 10], 0)
net.SGD(training_data, 20, 28, 0.5)

# HPT METHOD
# net2 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses, HPT_accuracies, stdpPRE, stdpPOST, stdpSYNAPSE = net2.HPT_SGD(training_data, 5, 28, 0.5, 1)

# COMBI RULE
# net3 = ANNScratchMNIST.Network([784, 30, 10], 0)
# net3.COMBI_SGD(training_data, 100, 28, 0.5, 0)

# HEBBIAN LEARNING RULES
# net.hebbian("imply", training_data, 20, 0.5)
# net.hebbian("competitive", training_data, 20, 0.5)

# METHOD COMPARISON ----------------------------------------------------------------------------------------------------
# horizontal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# # COST Comparison
# plt.plot(horizontal, losses)
# plt.plot(horizontal, HPT_losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()
#
# # ACC Comparison
# plt.plot(horizontal, accuracies)
# plt.plot(horizontal, HPT_accuracies)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

# Compare different HPT learning rates c -------------------------------------------------------------------------------
# horizontal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#
# net3 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses3, HPT_accuracies3, stdpPRE, stdpPOST, stdpSYNAPSE = net3.HPT_SGD(training_data, 20, 28, 0.5, 0.1)
#
# net4 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses4, HPT_accuracies4, stdpPRE, stdpPOST, stdpSYNAPSE = net4.HPT_SGD(training_data, 20, 28, 0.5, 0.5)
#
# net5 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses5, HPT_accuracies5, stdpPRE, stdpPOST, stdpSYNAPSE = net5.HPT_SGD(training_data, 20, 28, 0.5, 1)
#
# net6 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses6, HPT_accuracies6, stdpPRE, stdpPOST, stdpSYNAPSE = net6.HPT_SGD(training_data, 20, 28, 0.5, 5)
#
# net7 = ANNScratchMNIST.Network([784, 30, 10], 0)
# HPT_losses7, HPT_accuracies7, stdpPRE, stdpPOST, stdpSYNAPSE = net7.HPT_SGD(training_data, 20, 28, 0.5, 10)
#
# plt.plot(horizontal, HPT_losses3)
# plt.plot(horizontal, HPT_losses4)
# plt.plot(horizontal, HPT_losses5)
# plt.plot(horizontal, HPT_losses6)
# plt.plot(horizontal, HPT_losses7)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()
#
# plt.plot(horizontal, HPT_accuracies3)
# plt.plot(horizontal, HPT_accuracies4)
# plt.plot(horizontal, HPT_accuracies5)
# plt.plot(horizontal, HPT_accuracies6)
# plt.plot(horizontal, HPT_accuracies7)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

# SYNAPTIC CHANGE GRAPH ------------------------------------------------------------------------------------------------
# NOTE: for this, a [784, 30, 20, 10] network was used!
# colors = []
# for connection in stdpSYNAPSE:
#     if connection < 0:
#         colors.append(0)
#     elif connection >= 0:
#         colors.append(1)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(np.array(stdpPRE), np.array(stdpPOST), np.array(stdpSYNAPSE),
#            c=colors, cmap='coolwarm', alpha=0.6, zorder=0)
# ax.set_xlabel('Pre')
# ax.set_ylabel('Post')
# ax.set_zlabel('Synaptic Change')
# plt.show()

import MNISTloader
import ANNScratchMNIST
import matplotlib.pyplot as plt
import numpy as np

images, labels = MNISTloader.load_data(True)
training_data = [(images[i], labels[i]) for i in range(int(len(images)))]

font = {#'weight': 'bold',
        'size': 15}
plt.rc('font', **font)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

net = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
net.SGD(training_data, 20, 28, 0.5)

# HPT METHOD
net2 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
net2.HPT_SGD(training_data, 20, 28, 0.5, 0.5)

# COMBI RULE
# net3 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# net3.COMBI_SGD(training_data, 100, 28, 0.5, 0)

# HEBBIAN LEARNING RULES
# net = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# lossesImp, accuraciesImp = net.hebbian("imply", training_data, 20, 0.5)
# net2 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# lossesComp, accuraciesComp = net2.hebbian("competitive", training_data, 20, 0.5)
# horizontal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# # COST Comparison
# plt.plot(horizontal, lossesImp, label='Imply')
# plt.plot(horizontal, lossesComp, label='Competitive')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# # ACC Comparison
# plt.plot(horizontal, accuraciesImp, label='Imply')
# plt.plot(horizontal, accuraciesComp, label='Competitive')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

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
# net2 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses2, HPT_accuracies2, stdpPRE, stdpPOST, stdpSYNAPSE = net2.HPT_SGD(training_data, 20, 28, 0.5, 0)
#
# net3 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses3, HPT_accuracies3, stdpPRE, stdpPOST, stdpSYNAPSE = net3.HPT_SGD(training_data, 20, 28, 0.5, 0.1)
#
# net4 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses4, HPT_accuracies4, stdpPRE, stdpPOST, stdpSYNAPSE = net4.HPT_SGD(training_data, 20, 28, 0.5, 0.5)
#
# net5 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses5, HPT_accuracies5, stdpPRE, stdpPOST, stdpSYNAPSE = net5.HPT_SGD(training_data, 20, 28, 0.5, 1)
#
# net6 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses6, HPT_accuracies6, stdpPRE, stdpPOST, stdpSYNAPSE = net6.HPT_SGD(training_data, 20, 28, 0.5, 5)
#
# net7 = ANNScratchMNIST.Network([784, 30, 20, 10], 0)
# HPT_losses7, HPT_accuracies7, stdpPRE, stdpPOST, stdpSYNAPSE = net7.HPT_SGD(training_data, 20, 28, 0.5, 10)
#
# plt.plot(horizontal, HPT_losses2, label='c=0')
# plt.plot(horizontal, HPT_losses3, label='c=0.1')
# plt.plot(horizontal, HPT_losses4, label='c=0.5')
# plt.plot(horizontal, HPT_losses5, label='c=1')
# plt.plot(horizontal, HPT_losses6, label='c=5')
# plt.plot(horizontal, HPT_losses7, label='c=10')
# plt.xlabel('Epochs')#, fontweight='bold')
# plt.ylabel('Loss')#, fontweight='bold')
# plt.legend(prop={'size': 12})
# plt.savefig('highresLossHPT.png', dpi=300)
# plt.show()
#
# plt.plot(horizontal, HPT_accuracies2, label='c=0')
# plt.plot(horizontal, HPT_accuracies3, label='c=0.1')
# plt.plot(horizontal, HPT_accuracies4, label='c=0.5')
# plt.plot(horizontal, HPT_accuracies5, label='c=1')
# plt.plot(horizontal, HPT_accuracies6, label='c=5')
# plt.plot(horizontal, HPT_accuracies7, label='c=10')
# plt.xlabel('Epochs')#, fontweight='bold')
# plt.ylabel('Accuracy')#, fontweight='bold')
# plt.legend(prop={'size': 12})
# plt.savefig('highresAccuracyHPT.png', dpi=300)
# plt.show()

# SYNAPTIC CHANGE GRAPH ------------------------------------------------------------------------------------------------
# colors = []
# for connection in stdpSYNAPSE:
#     if connection < 0:
#         colors.append(0)
#     elif connection >= 0:
#         colors.append(1)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# dots = ax.scatter(np.array(stdpPRE), np.array(stdpPOST), np.array(stdpSYNAPSE),
#            c=colors, cmap='coolwarm', alpha=0.6, zorder=0)
# ax.set_xlabel('Pre')
# ax.set_ylabel('Post')
# ax.set_zlabel('Synaptic Change')
# ax.set_zlim3d((-0.001, 0.005))
# plt.show()
#
# plt.scatter(np.array(stdpPRE), np.array(stdpPOST), c=colors, cmap='coolwarm', alpha=0.6, zorder=0)
# plt.xlabel('Pre')
# plt.ylabel('Post')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
import util as u
import data as d

class MLP():

    # initialization
    def __init__(self, inputs, hidden_layer_perceptrons, output_neurons):
        self.perceptrons = []
        # add hidden layer
        self.perceptrons.append([Perceptron(inputs) for x in range(hidden_layer_perceptrons)])
        # add output layer
        self.perceptrons.append([Perceptron(hidden_layer_perceptrons) for x in range(output_neurons)])

    # forward step
    def forward_step(self, inputs):
        self.outputs = []
        self.outputs.append(inputs) # needed for the backpropagation step

        # put the input in one layer and compute the output
        # this output is then used as the input for the next layer
        for l in range(len(self.perceptrons)):
            outputs = np.array([p.forward_step(self.outputs[-1]) for p in self.perceptrons[l]])
            self.outputs.append(outputs)

    # backpropagation step
    def backprop_step(self, target):
        delta = None
        deltas = {}

        # start in last layer, perform backpropagation
        for i, l in enumerate(reversed(self.perceptrons)):
            # create an entry for the deltas of the layer
            deltas[len(self.perceptrons)-i-1] = []

            # go through each Perceptron of the layer
            for n_perceptrons, p in enumerate(l):

                # last layer: calculate delta with target (the output of the last layer), and derivation of input sum in Perceptron
                if delta is None:
                    delta = -(target-self.outputs[-1-i])*u.sigmoidprime(p.get_input())

                # hidden layers: calculate delta with last delta, the weights of the next layer connected to the current neuron and the derivative of the input signal
                else:
                    delta = np.sum(delta * np.array([next_layer_perceptron.get_weights()[n_perceptrons] for next_layer_perceptron in self.perceptrons[-i]])) * u.sigmoidprime(p.get_input())
                # append the deltas of the hole layer to the dict
                deltas[len(self.perceptrons)-i-1].append(delta)

        # update the weights with the computed deltas
        for l, l_deltas in deltas.items():
            for p, delta in zip(self.perceptrons[l], l_deltas):
                p.update(delta)

    # training of the mlp
    def train(self, data, epochs=100, info_all_n_epochs=100, accuracy_over_last_n_epochs=100):
        self.all_loss = []
        self.all_accuracy = []
        self.epochs = epochs
        self.correct = []
        for epoch in range(1, epochs + 1):
            # go through all samples
            for sample in data:
                target = sample[2] # target = last entry in sample
                self.forward_step(sample[:2]) # first two entries = input data
                self.backprop_step(target)
                self.all_loss.append(self.loss(target))
                self.all_accuracy.append(self.accuracy(target, accuracy_over_last_n_epochs))

            if epoch % info_all_n_epochs == 0:
                print("Epoch {}:".format(epoch))
                print("Loss: {}".format(np.mean(self.all_loss)))
                print("Accuracy: {}".format(np.mean(self.all_accuracy)))
                print("")

    # visualization
    def visualization(self):
        # plot training
        fig, ax = plt.subplots(2)
        ax[0].plot(range(0, self.epochs),
                    self.smooth(np.array(self.all_loss).squeeze(), 10)[::4])
        ax[0].set(xlabel='epochs', ylabel='loss',
                   title='Training loss')
        ax[0].grid()

        ax[1].plot(range(0, self.epochs),
                    self.smooth(np.array(self.all_accuracy).squeeze(), 1)[::4])
        ax[1].set(xlabel='epochs', ylabel='accuracy',
                   title='Training accuracy')
        ax[1].grid()
        plt.show()

    # smoothens the plot
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # the loss function (t − y)¹
    def loss(self, target):
        return (target - self.outputs[-1])**2

    # computation of the accuracy
    def accuracy(self, target, accuracy_over_last_n_epochs):
        self.correct.append(np.round(self.outputs[-1]) == target)
        if len(self.correct) <= accuracy_over_last_n_epochs:
            return np.mean(self.correct)
        else:
            return np.mean(self.correct[len(self.correct)-accuracy_over_last_n_epochs:])


# initialize MLP, load data, train and visualize
mlp = MLP(inputs=2, hidden_layer_perceptrons=4, output_neurons=1)
data = d.load_data_and()
mlp.train(data, epochs=1000, info_all_n_epochs=100, accuracy_over_last_n_epochs=30)
mlp.visualization()

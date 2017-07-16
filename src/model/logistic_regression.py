# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import random

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=30, num_layers=1, minibatch_size = 1):

        self.learningRate = learningRate
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.layers = []
        for i in xrange(num_layers):
            isClassifier = False
            if (i == num_layers - 1) : isClassifier = True
            self.layers.append(LogisticLayer(nIn=self.trainingSet.input.shape[1], nOut=1, activation='sigmoid',  isClassifierLayer=isClassifier))


    def train(self, verbose=False):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import DifferentError

        learned = False
        iteration = 0

        training_pairs = zip(self.trainingSet.input, self.trainingSet.label)

        while not learned:
            random.shuffle(training_pairs)
            minibatches = [training_pairs[j:j+self.minibatch_size] for j in xrange(0, len(training_pairs), self.minibatch_size)]
            for minibatch in minibatches:
                self.updateMinibatch(minibatch)

            iteration += 1

            if iteration >= self.epochs:
                # stop criteria is reached
                learned = True


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) > 0.5


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))


    def feedforward(self, input):
        for layer in self.layers :
            input = layer.forward(input)
        return input


    def updateWeights(self):
        for layer in self.layers:
            layer.updateWeights(learning_rate=self.learningRate, minibatch_size=self.minibatch_size)


    def updateMinibatch(self, minibatch):
        nextWeights = 0
        grad = 0
        for input_x, label_x in minibatch :
            o_x = input_x
            for layer in self.layers :
                layer.input[1:] = o_x[:, None]
                o_x = layer.forward(o_x)

            cost_derivative = o_x-label_x

            for layer in reversed(self.layers) :
                cost_derivative = layer.computeDerivative(nextDerivatives=cost_derivative, nextWeights=nextWeights)
                nextWeights = layer.weights

            self.updateWeights()


    def fire(self, input):
        return self.feedforward(input)

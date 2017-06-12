# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from model.classifier import Classifier
from util.loss_functions import DifferentError
from util.loss_functions import MeanSquaredError

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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        lossDiffer = DifferentError()
        lossMSE = MeanSquaredError()
        list_of_errors = []
        gradient = 0
        for epoch in xrange(self.epochs):
            outputs = []
            for input, target in zip(self.trainingSet.input, self.trainingSet.label):
                output = self.fire(input)
                outputs.append(output)
                error = lossDiffer.calculateError(target, output)
                gradient = gradient + error * input
            self.updateWeights(gradient)

            mse = lossMSE.calculateError(self.trainingSet.label, outputs)
            list_of_errors.append(mse)

            #if verbose :
                #logging.info("Epoch : %i; MSE : %f", epoch + 1, mse)

        return list_of_errors, range(self.epochs+1)


        
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
        return int(self.fire(testInstance))

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

    def updateWeights(self, grad):
        self.weight = self.weight + self.learningRate * grad


    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))

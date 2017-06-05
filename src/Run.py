#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myLogisticNeuron1 = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    myLogisticNeuron2 = LogisticRegression(data.trainingSet,
                                          data.validationSet,
                                          data.testSet,
                                          learningRate=0.005,
                                          epochs=60)
    myLogisticNeuron3 = LogisticRegression(data.trainingSet,
                                          data.validationSet,
                                          data.testSet,
                                          learningRate=0.005,
                                          epochs=100)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nFirst Neuron has been training..")
    listErrors1, epochs1 = myLogisticNeuron1.train()
    print("Done..")

    print("\nSecond Neuron has been training..")
    listErrors2, epochs2 = myLogisticNeuron2.train()
    print("Done..")

    print("\nThird Neuron has been training..")
    listErrors3, epochs3 = myLogisticNeuron3.train()
    print("Done..")

    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    logisticPred1 = myLogisticNeuron1.evaluate()
    logisticPred2 = myLogisticNeuron2.evaluate()
    logisticPred3 = myLogisticNeuron3.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the first neuron recognizer:")
    #evaluator.printComparison(data.testSet, logisticPred)
    evaluator.printAccuracy(data.testSet, logisticPred1)

    logisticPred = myLogisticNeuron2.evaluate()

    print("\nResult of the second neuron recognizer:")
    # evaluator.printComparison(data.testSet, logisticPred)
    evaluator.printAccuracy(data.testSet, logisticPred2)

    print("\nResult of the third neuron recognizer:")
    # evaluator.printComparison(data.testSet, logisticPred)
    evaluator.printAccuracy(data.testSet, logisticPred3)

    plt.plot(epochs1[1:], listErrors1, 'r', label = '30 epochs')
    plt.plot(epochs2[1:], listErrors2, 'b', label = '60 epochs')
    plt.plot(epochs3[1:], listErrors3, 'g', label = '100 epochs')
    plt.legend(loc='upper left')

    plt.show()




    
    
if __name__ == '__main__':
    main()

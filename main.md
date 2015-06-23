# Parity---Random-Forest
# Can a random forest ML algorithm learn mod?
# Main file

#Psuedocode:
#Generate master data list (1 to 1000 without 100 numbers) in a list
#Randomly select 450 numbers each time and grow 100 classification trees
#Evaluate accuracy of forest by testing a number on trees that did not use that number to grow
#(not part of code) Conduct hypothesis test to determine
#whether model does better than chance

from randomTree import *
from getTrainingNums import *
from numpy import *
import random

#Main
def main():
    trainingSet, testingSet = getTrainingNums.createFiles()
    forest = [] #list of trees
    
    #Making a forest
    #"Randomizing features" does not apply because we only have one feature
    for i in range(100):
        treeBuilder = random.sample(trainingSet,450) #Takes a random sample from the training set
        treeBuilder = treeBuilder.sort()
        numMat = randomTree.loadDataSet(treeBuilder) #Puts the data in a matrix
        tree = randomTree.createTree(numMat)
        forest.append(tree)
        
    #Testing the forest
    for i in range(len(testingSet)):
        #Psuedocode: (because I don't know how to code this)
        #Push testingSet[i] down each tree in the forest
        #Print the parity associated with the leaf it reaches
        #Compare with actual answers
        #(not in code) do a hypothesis test

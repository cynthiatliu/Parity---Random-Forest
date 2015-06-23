# Parity---Random-Forest
# Can a random forest ML algorithm learn mod?
# Random tree generation

from numpy import *
#[:] means all of a list, so [:,n] means all the (n+1)th values in all rows

#----------------------------------------------------------------

#Returns the parity associated with the leaf
def ranLeaf(dataSet):
    classList = [example[-1] for example in dataSet]
    labelcounts = {}
    for item in classList: #even or odd
        if item not in labelcounts.keys():
            labelcounts[item] = 0
        labelcounts[item] += 1
        
    #Now we can get a "majority vote" with 2 classes
    toReturn = "even" #Default
    for key in labelcounts:
        prob = float(labelcounts[key])/len(classList)
        if prob > .500000: #A majority
            toReturn = key
    
    return toReturn

#----------------------------------------------------------------

#Returns the error of the leaf node
def ranErr(dataSet):
    parity = ranLeaf(dataSet)
    mistakes = 0
    for num in dataSet:
        mod = num%2
        if (mod == 0): modWord = "even"
        else: modWord = "odd"
        
        if parity != modWord: mistakes += 1
    return mistakes

#----------------------------------------------------------------

#Read the data, map them to floats, then add them to a matrix and return the matrix
#Observe that we do not break the target variable into a separate list
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        linArr = line.strip().split('\t')
        strLine = map(string, linArr)
        dataMat.append(fltLine)
    return dataMat

#-----------------------------------------------------------------

#Splitting the data set by a feature's value (feature represented by its index)
def binSplitDataSet(dataSet, index):
    mat0 = dataSet[nonzero(dataSet[index:])[0],:][0] #data is presorted
    mat1 = dataSet[nonzero(dataSet[0:index])[0],:][0]
    return mat0, mat1

#-----------------------------------------------------------------

#Chooses the best value on which to make a split (greedy algorithm)
def chooseBestSplit(dataSet, leafType=ranLeaf, errType=ranErr, ops=(3,4)):
    tolS = ops[0]; tolN = ops[1] #Tolerance in error and size (we do not want arbitrary accuracy --> overfitting)

    #Check the number of unique values by creating a set of all target variables
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet) #exit if all the target variable values are equal, no use in splitting
    
    labelcounts = {}
    for item in classList: #even or odd
        if item not in labelcounts.keys():
            labelcounts[item] = 0
        labelcounts[item] += 1        
    
    #Determine which partition produces both (in econ terms) absolute and comparative advantage
    #Not sure if we are guaranteed such a partition
    #Best idea right now is to loop through with a partition index and look at every set, if the assumption is verifie

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestIndex

#------------------------------------------------------------------

#Making a tree
def createTree(dataSet, leafType=ranLeaf, errType=ranErr, ops=(3,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val #Return a node if stopping condition, that there is no good split, is met
    
    retTree = {} #initiating a tree
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    
    retTree['left'] = createTree(lSet, leafType, errType, ops) #Recursion
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree    

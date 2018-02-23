
# coding: utf-8

# In[53]:


import os
import sys
import numpy as np
SEPARATOR = '******######******######******######******######******'


# In[54]:


def readModelParameters():
    global SEPARATOR
    transitionMatrix = []
    emissionMatrix = []
    initialProbablities = {}
    endProbablities = {}
    tagIndexDict = {}
    wordIndexDict = {}
    filename = "hmmmodel.txt"
    fileContents = []
    uniqueWords = []
    with open(filename, 'r') as f:
        fileContents = f.readlines()
    lenUniqueTags = fileContents[0]
    lenUniqueWords = fileContents[1]
    
    flag = 0
    
    for index in range(0, len(fileContents), 1):
        line = fileContents[index].strip()
        
        if line == SEPARATOR:
            flag += 1
            start = 0
            continue
        
        if flag == 1:
            if start == 0:
                start = 1
                continue
            transitionMatrix.append(map(float, line.split(",")))
        elif flag == 2:
            if start == 0:
                start = 1
                continue
            emissionMatrix.append(map(float, line.split(",")))
        elif flag == 3:
            if start == 0:
                start = 1
                continue
            dicentry = line.split('\t')
            try:
                initialProbablities[dicentry[0]] = float(dicentry[1])
            except:
                pass
        elif flag == 4:
            if start == 0:
                start = 1
                continue
            dicentry = line.split('\t')
            try:
                endProbablities[dicentry[0]] = float(dicentry[1])
                tagIndexDict[dicentry[0]] = int(dicentry[2])
            except:
                pass
        elif flag == 5:
            if start == 0:
                start = 1
                continue
            uniqueTags = line.split("\t")
        elif flag == 6:
            if start == 0:
                start = 1
                continue
            mostFrequentTag = line
        elif flag == 7:
            if start == 0:
                start = 1
                continue
            numTag = line
        elif flag == 8:
            if start == 0:
                start = 1
                continue
            uniqueWordsAndIndex = line.split("\t")
            try:
                uniqueWords.append(uniqueWordsAndIndex[0])
                wordIndexDict[uniqueWordsAndIndex[0]] = int(uniqueWordsAndIndex[1])
            except:
                uniqueWords.append(" ")
                wordIndexDict[" "] = int(uniqueWordsAndIndex[0])
        else:
            pass
    
    return transitionMatrix , emissionMatrix, initialProbablities, endProbablities, uniqueTags, tagIndexDict, uniqueWords, wordIndexDict, mostFrequentTag, numTag


# In[55]:


def getTestDataFromFile():
    filenameTestRaw = sys.argv[1]
    fileOpenTest = open(filenameTestRaw,"r")
    testData = fileOpenTest.readlines()
    return testData

def getPrediction(testData):
    global transitionMatrix, emissionMatrix
    prediction = ""
    for sentenceIndex in range(len(testData)):
        words = testData[sentenceIndex].strip("\n").split(" ")
        predictedSentence = viterbiMatrix(words)
        prediction += predictedSentence + "\n"
    
    filename = 'hmmoutput.txt'    
    with open(filename, 'w') as f:
        f.write(prediction)
    
        
    


def viterbiMatrix(words):
    global uniqueTags, tagIndexDictReverse, transitionMatrix, emissionMatrix, tagIndexDict, mostFrequentTag, numTag
    totalWordCount = 0
    correctlyTagged = 0
    uniqueTagsLength = len(uniqueTags)
    sentenceLength = len(words)

        
    
    viterbiMatrix = np.zeros(shape = (uniqueTagsLength, sentenceLength))
    #viterbiMatrix += np.finfo(float).min
    
    correspondingTags = [[mostFrequentTag for y in range(sentenceLength)]for x in range(uniqueTagsLength)]
    for col in range(sentenceLength):
        for row in range(uniqueTagsLength):
            tag = tagIndexDictReverse[row]
            word = words[col]
            try:
                obsProbability = emissionMatrix[row][wordIndexDict[word]]
                if obsProbability == 0.0:
                    continue
            except KeyError as e:
                try:
                    obsProbability = emissionMatrix[row][wordIndexDict[word.lower()]]
                except:
                    obsProbability = 1.0
                    numCount = 0
                    for character in word:
                        if character.isdigit():
                            numCount += 1
                    if numCount * 1.0 > 0.6 * len(word):
                        if tag == numTag:
                            obsProbability = 1.0
                        else:
                            obsProbability = 0.000000000000000001

                                
            if col == 0:
                transitionProbability = initialProbablities[tag]
                viterbiMatrix[row][col] = transitionProbability * obsProbability
                correspondingTags[row][col] = tag
            else:
                for prevColrow in range(uniqueTagsLength):
                    prevVal = viterbiMatrix[prevColrow][col-1]
                    if prevVal == 0.0:
                        continue
                    prevTag = tagIndexDictReverse[prevColrow]
                    transitionProbability = transitionMatrix[prevColrow][row]
                    probabilityVal = transitionProbability * obsProbability * prevVal
                    if probabilityVal > viterbiMatrix[row][col]:
                        correspondingTags[row][col] = prevTag
                        viterbiMatrix[row][col] = probabilityVal
                                            

        
    finalStateVal = 0.0
    finalTag = mostFrequentTag
    for rowIndex in range(uniqueTagsLength):
        tag = tagIndexDictReverse[rowIndex]
        finalStateProbability = endProbablities[tag] * viterbiMatrix[rowIndex][sentenceLength - 1]
        if finalStateProbability > finalStateVal:
            finalTag = tag
            finalStateVal = finalStateProbability
            
    assignedTags = [finalTag]    
    currentTag = finalTag
    for colIndex in range(sentenceLength -1,0,-1):
        try:
            currentTag = correspondingTags[tagIndexDict[currentTag]][colIndex]
        except KeyError as e:
            pass
        assignedTags.append(currentTag)

    assignedTags = assignedTags[::-1]

    predictedWordTagPair = []
    for i in range(sentenceLength):
        predictedWordTagPair.append('/'.join([words[i], assignedTags[i]]))
    return ' '.join(predictedWordTagPair)


# In[56]:


transitionMatrix , emissionMatrix, initialProbablities, endProbablities, uniqueTags, tagIndexDict, uniqueWords, wordIndexDict, mostFrequentTag, numTag = readModelParameters()


# In[57]:


tagIndexDictReverse = {}
for tag, index in tagIndexDict.iteritems():
    tagIndexDictReverse[index] = tag
    


# In[58]:


testData = getTestDataFromFile()
getPrediction(testData)


# In[ ]:





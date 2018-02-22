
# coding: utf-8

# In[13]:


import os
import sys
import numpy as np
SEPARATOR = '******######******######******######******######******'


# In[14]:


def readDatafromFile():
    filename = sys.argv[1]
    fileOpen = open(filename,"r")
    fileData = fileOpen.readlines()
    return fileData


# In[15]:


fileData = readDatafromFile()


# In[16]:


def getTagWordfromPair(wordTagPair):
    tag = wordTagPair.split("/")[-1]
    wordList = wordTagPair.split("/")[:-1]
    word = '/'.join(wordList)
    return tag,word


# In[17]:


def getTotalWordTagOccurrences(fileData):
    totalWordOccurrences = {}
    totalTagOccurrences = {}
    numberTagCount = {}
    for line in fileData:
        wordTagPair = line.strip("\n").split(" ")
        for ele in wordTagPair:
            tag,word = getTagWordfromPair(ele)
            numCount = 0
            for character in word:
                if character.isdigit() == True:
                    numCount += 1
            if numCount * 1.0 > 0.6 * len(word):
                try:
                    numberTagCount[tag] += 1
                except:
                    numberTagCount[tag] = 1


            try:
                totalWordOccurrences[word] += 1
            except KeyError as e:
                totalWordOccurrences[word] = 1

            try:
                totalTagOccurrences[tag] += 1
            except KeyError as e:
                totalTagOccurrences[tag] = 1
    
    maxNumTag = max(numberTagCount, key = numberTagCount.get)
    return totalTagOccurrences, totalWordOccurrences, maxNumTag

def getUniqueTags(totalTagOccurrences):
    uniqueTags = totalTagOccurrences.keys()
    uniqueTags.sort()
    return uniqueTags


# In[18]:


def getTagDictonaries(uniqueTags):
    tagIndexDict = {}
    tagIndexDictReverse = {}
    for tagIndex, tag in enumerate(uniqueTags):
        tagIndexDict[tag] = tagIndex
        tagIndexDictReverse[tagIndex] = tag
    return tagIndexDict, tagIndexDictReverse


# In[19]:


def getUniqueWords(totalWordOccurrences):
    uniqueWords = totalWordOccurrences.keys()
    uniqueWords.sort()
    return uniqueWords

def getWordDictionaries(uniqueWords):
    wordIndexDict = {}
    wordIndexDictReverse = {}
    for wordIndex, word in enumerate(uniqueWords):
        wordIndexDict[word] = wordIndex
        wordIndexDictReverse[wordIndex] = word
    return wordIndexDict, wordIndexDictReverse


# In[20]:


totalTagOccurrences, totalWordOccurrences, maxNumTag = getTotalWordTagOccurrences(fileData)
uniqueTags = getUniqueTags(totalTagOccurrences)
uniqueWords = getUniqueWords(totalWordOccurrences)
tagIndexDict, tagIndexDictReverse = getTagDictonaries(uniqueTags)
wordIndexDict, wordIndexDictReverse = getWordDictionaries(uniqueWords)


# In[21]:


def getTransitionMatrix(fileData, uniqueTags, tagIndexDict):
    transitionMatrix = np.ones(shape=(len(uniqueTags), len(uniqueTags)))
    
    for line in fileData:
        wordTagPairs = line.strip("\n").split(" ")
        for pairIndex in range(len(wordTagPairs) - 1):
            tag1 = wordTagPairs[pairIndex].split("/")[-1]
            tag2 = wordTagPairs[pairIndex + 1].split("/")[-1]
            tag1Index = tagIndexDict[tag1]
            tag2Index = tagIndexDict[tag2]
            transitionMatrix[tag1Index][tag2Index] += 1
    
    transitionMatrix = transitionMatrix / transitionMatrix.sum(axis = 1, keepdims = True)
    #transitionMatrix = np.log(transitionMatrix)
    return transitionMatrix


# In[22]:


def getInitialProbabilities(fileData, uniqueTags):
    initialProbablities = {}
    sentenceCount = len(fileData)
    for line in fileData:
        wordTagPairs = line.strip("\n").split(" ")
        tag = wordTagPairs[0].split("/")[-1]
        try:
            initialProbablities[tag] += 1
        except KeyError as e:
            initialProbablities[tag] = 1
        
    
        
    for tag in uniqueTags:
        try:
            initialProbablities[tag] += 0.000000001
        except KeyError as e:
            initialProbablities[tag] = 0.000000001
        sentenceCount += 0.000000001
        
    initialProbablities.update((tag, (value*1.0/sentenceCount) ) for tag, value in initialProbablities.items())
    return initialProbablities


# In[23]:


def getEndProbabilities(fileData, uniqueTags):
    endProbablities = {}
    sentenceCount = len(fileData)
    for line in fileData:
        lastWordTagPair = line.strip("\n").split(" ")[-1]
        tag = lastWordTagPair.split("/")[-1]
        try:
            endProbablities[tag] += 1
        except KeyError as e:
            endProbablities[tag] = 1



    for tag in uniqueTags:
        try:
            endProbablities[tag] += 0.000000001
        except KeyError as e:
            endProbablities[tag] = 0.000000001
        sentenceCount += 0.000000001

    endProbablities.update((tag, (value*1.0/sentenceCount) ) for tag, value in endProbablities.items())
    return endProbablities


# In[24]:


def getEmissionMatrix(fileData):
    global uniqueTags, uniqueWords, tagIndexDict, wordIndexDict
    emissionMatrix = np.zeros(shape = (len(uniqueTags), len(uniqueWords)))

    for line in fileData:
        wordTagPairs = line.strip("\n").split(" ")
        for ele in wordTagPairs:
            tag, word = getTagWordfromPair(ele)

            tagIndex = tagIndexDict[tag]
            wordIndex = wordIndexDict[word]

            emissionMatrix[tagIndex][wordIndex] += 1
    #emissionMatrix += 0.000000000000001
    emissionMatrix = emissionMatrix / emissionMatrix.sum(axis = 1, keepdims = True)
    #emissionMatrix = np.log(emissionMatrix)
    return emissionMatrix


# In[25]:


def writeModelParameters(transitionMatrix, emissionMatrix, initialProbablities, endProbablities, uniqueTags, uniqueWords, tagIndexDict, wordIndexDict, maxNumTag, filename='hmmmodel.txt'):
    global SEPARATOR, totalTagOccurrences
#     output = "Transition Matrix:\n" + str(transitionMatrix) + "\n\nEmission Matrix:\n" + str(emissionMatrix) + "\n\nInitial Probabilities:\n" + str(initialProbablities) + "\n\nEnd Probabilities:\n" + str(endProbablities);
    output = str(len(uniqueTags)) + "\n" +  str(len(uniqueWords)) + "\n"
    output += SEPARATOR + '\n'
    output += "Transition Matrix:" + '\n'
    transRows = ""
    for row in transitionMatrix:
        transRows += ','.join(map(str,row)) + "\n"
        
    output += transRows
    output += SEPARATOR + '\n'
        
    output += "Emission Matrix:" + '\n'
    emiRows = ""
    for row in emissionMatrix:
        emiRows += ','.join(map(str,row)) + "\n"
        
    output += emiRows
    
    output += SEPARATOR + '\n'
    output += "Initial Probabilities:" + '\n'
    
    initialProbab = ""
    for key in initialProbablities:
        initialProbab += key + "\t" + str(initialProbablities[key]) + "\n"
    
    output += initialProbab
    output += SEPARATOR + '\n'
    
    output += "End Probabilities:" + '\n'
        
    endProbab = ""
    for key in endProbablities:
        endProbab += key + "\t" + str(endProbablities[key]) + "\t" + str(tagIndexDict[key]) + "\n"
        
    output += endProbab
    output += SEPARATOR + '\n'
    
    output += "Unique Tags:" + '\n'
    tags = '\t'.join(uniqueTags) + '\n'
    
    output += tags
    
    output += SEPARATOR + '\n'

    output += "Most Frequent Tag:" + '\n'

    output += max(totalTagOccurrences, key = totalTagOccurrences.get) + '\n'

    output += SEPARATOR + '\n'

    output += "Number Tag:" + '\n'

    output += maxNumTag + '\n'

    output += SEPARATOR + '\n'
    
    words = ""
    for word in uniqueWords:
        wordAndIndex = word + "\t" + str(wordIndexDict[word])
        words += wordAndIndex + '\n'
    
    output += "Unique Words:" + '\n'
    output += words
    
    with open (filename,'w') as f:
        f.write(output)


# In[ ]:
transitionMatrix = getTransitionMatrix(fileData, uniqueTags, tagIndexDict)
emissionMatrix = getEmissionMatrix(fileData)
emissionMatrixNormal = emissionMatrix.tolist()
initialProbablities = getInitialProbabilities(fileData, uniqueTags)
endProbablities = getEndProbabilities(fileData, uniqueTags)
writeModelParameters(transitionMatrix, emissionMatrixNormal, initialProbablities, endProbablities, uniqueTags, uniqueWords,tagIndexDict, wordIndexDict, maxNumTag)




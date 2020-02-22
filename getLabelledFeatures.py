import pickle
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

def getStem(word, stemmer):
    return stemmer.stem(word)
    
def getShape(word):
    shape = ""

    for char in word:
        if char.isupper():
            shape += 'X'
        else:
            shape += 'x'

    return shape

def getPOSTag(line, wordNum):
    words = []
    for wordLabelPair in line:
        words.append(wordLabelPair[0])

    tags = pos_tag(words)

    return tags[wordNum][1]

def getPrevTokensClasses(line, wordNum):
    if wordNum > 2:
        prev_tokens = [line[wordNum - 3][0], line[wordNum - 2][0], line[wordNum - 1][0]]
        prev_classes = [line[wordNum - 3][1], line[wordNum - 2][1], line[wordNum - 1][1]]

    elif wordNum > 1:
        prev_tokens = ["", line[wordNum - 2][0], line[wordNum - 1][0]]
        prev_classes = ["", line[wordNum - 2][1], line[wordNum - 1][1]]

    elif wordNum > 0:
        prev_tokens = ["", "", line[wordNum - 1][0]]
        prev_classes = ["", "", line[wordNum - 1][1]]

    else:
        prev_tokens = ["", "", ""]
        prev_classes = ["", "", ""]

    for i in [0,1,2]:
        prev_tokens[i] = prev_tokens[i].lower()

    return prev_tokens, prev_classes

def getNextTokens(line, wordNum):
    if len(line) - wordNum > 3:
        next_tokens = [line[wordNum + 1][0], line[wordNum + 2][0], line[wordNum + 3][0]]
    elif len(line) - wordNum > 2:
        next_tokens = [line[wordNum + 1][0], line[wordNum + 2][0], ""]
    elif len(line) - wordNum > 1:
        next_tokens = [line[wordNum + 1][0], "", ""]
    else:
        next_tokens = ["", "", ""]

    for i in [0,1,2]:
        next_tokens[i] = next_tokens[i].lower()

    return next_tokens


inputFile = open("labelledSentences.pickle","rb")
inputList = pickle.load(inputFile)
inputFile.close()

outputFile = open("labelledFeatures.txt", "w+")

stemmer = PorterStemmer()

###############################################################
# Is removing stopwords a good idea?!
###############################################################
stopList = stopwords.words('english') + list(string.punctuation)

allFeatures = []
for line in inputList:
    for wordNum in range(len(line)):
        if line[wordNum][0] not in stopList:
            # set up feature class tuple
            features = {'token':line[wordNum][0].lower()}
            classLabel = line[wordNum][1]

            # add new features
            # stemmed token
            features['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)

            # token shape
            features['shape'] = getShape(line[wordNum][0])

            # part of speech tag
            features['pos_tag'] = getPOSTag(line, wordNum)

            # Previous 3 tokens
            # Previous 3 tokens classes
            prev_tokens_classes = getPrevTokensClasses(line, wordNum)
            features['prev_tokens'], features['prev_classes'] = str(prev_tokens_classes[0]), str(prev_tokens_classes[1])

            # Next 3 tokens
            features['next_tokens'] = str(getNextTokens(line, wordNum))

            # save feature class tuple
            outputFile.write(str((features, classLabel)) + "\n")
            allFeatures.append((features, classLabel))

with open('labelledFeatures.pickle', 'wb+') as f:
    pickle.dump(allFeatures, f)

inputFile.close()
outputFile.close()
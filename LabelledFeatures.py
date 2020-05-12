import pickle
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import re

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

def getPrevTokensClasses(line, wordNum, stopList):
    prev_tokens = []
    prev_classes = []
    for i in range(1, wordNum + 1):
        if len(prev_tokens) < 3 and line[wordNum - i][0] not in stopList:
            prev_tokens.append(line[wordNum - i][0])
            prev_classes.append(line[wordNum - i][1])

    while len(prev_tokens) < 3:
        prev_tokens.append("")
        prev_classes.append("")

    for i in [0,1,2]:
        prev_tokens[i] = prev_tokens[i].lower()

    return prev_tokens, prev_classes

def getNextTokens(line, wordNum, stopList):
    next_tokens = []
    for i in range(1, len(line) - wordNum):
        if len(next_tokens) < 3 and line[wordNum + i][0] not in stopList:
            next_tokens.append(line[wordNum + i][0])

    while len(next_tokens) < 3:
        next_tokens.append("")

    for i in [0,1,2]:
        next_tokens[i] = next_tokens[i].lower()

    return next_tokens


def getLabelledFeatures(labelled_sentences_file, labelled_features_file):
    inputFile = open(labelled_sentences_file[:-4] + ".pickle","rb")
    inputList = pickle.load(inputFile)
    inputFile.close()

    outputFile = open(labelled_features_file[:-4] + ".txt", "w+", encoding="utf-8")
    outputFile2 = open(labelled_features_file[:-4] + "2.txt", "w+", encoding="utf-8")
    outputFile3 = open(labelled_features_file[:-4] + "3.txt", "w+", encoding="utf-8")
    outputFile4 = open(labelled_features_file[:-4] + "4.txt", "w+", encoding="utf-8")
    outputFile5 = open(labelled_features_file[:-4] + "5.txt", "w+", encoding="utf-8")
    outputFile6 = open(labelled_features_file[:-4] + "6.txt", "w+", encoding="utf-8")
    outputFile7 = open(labelled_features_file[:-4] + "7.txt", "w+", encoding="utf-8")
    outputFile8 = open(labelled_features_file[:-4] + "8.txt", "w+", encoding="utf-8")

    stemmer = PorterStemmer()
    stops = stopwords.words('english')
    stops.remove('m')
    stopList = stops + list(string.punctuation)
    stopList.remove("'")
    stopList.remove('"')

    numbers = ['0','1','2','3','4','5','6','7','8','9']
    units = ['m', 'cm', 'mm', 'ft', 'in', 'inches', 'feet', 'foot', 'km', 'miles', 'kilometre', 'kilometres', 'centimetre', 'metre', 'centimetres', 'metres']

    allFeatures = []
    allFeatures2 = []
    allFeatures3 = []
    allFeatures4 = []
    allFeatures5 = []
    allFeatures6 = []
    allFeatures7 = []
    allFeatures8 = []
    for line in inputList:
        for wordNum in range(len(line)):
            if line[wordNum][0] not in stopList:
                # set up feature class tuple
                features = {'token':line[wordNum][0].lower()}
                features2 = {'token':line[wordNum][0].lower()}
                features3 = {'token':line[wordNum][0].lower()}
                features4 = {'token':line[wordNum][0].lower()}
                classLabel = line[wordNum][1]

                # add new features
                # stemmed token
                features['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)
                features3['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)
                features4['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)

                # token shape
                features['shape'] = getShape(line[wordNum][0])
                features2['shape'] = getShape(line[wordNum][0])
                features3['shape'] = getShape(line[wordNum][0])
                features4['shape'] = getShape(line[wordNum][0])

                # part of speech tag
                features['pos_tag'] = getPOSTag(line, wordNum)
                features2['pos_tag'] = getPOSTag(line, wordNum)
                features3['pos_tag'] = getPOSTag(line, wordNum)
                features4['pos_tag'] = getPOSTag(line, wordNum)

                # Previous 3 tokens
                # Previous 3 tokens classes
                prev_tokens_classes = getPrevTokensClasses(line, wordNum, stopList)
                features['prev_tokens'], features['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()
                features2['prev_tokens'], features2['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()
                features4['prev_tokens'], features4['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()

                # Next 3 tokens
                features['next_tokens'] = str(getNextTokens(line, wordNum, stopList))
                features2['next_tokens'] = str(getNextTokens(line, wordNum, stopList))
                features4['next_tokens'] = str(getNextTokens(line, wordNum, stopList))

                # Sub-tokens
                features['sub-tokens'] = "('','')"
                features2['sub-tokens'] = "('','')"
                features3['sub-tokens'] = "('','')"
                # Contains numbers
                features['numbers'] = False
                features2['numbers'] = False
                features3['numbers'] = False
                for c in range(len(line[wordNum][0])):
                    if line[wordNum][0][c] in numbers:
                        features['numbers'] = True
                        features2['numbers'] = True
                        features3['numbers'] = True
                        try:
                            if c != 0 and line[wordNum][0][c + 1] not in numbers:
                                features['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                                features2['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                                features3['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                        except IndexError:
                            pass

                # Contains a unit of size
                features['unit_of_size'] = False
                features2['unit_of_size'] = False
                features3['unit_of_size'] = False
                for unit in units:
                    if unit in line[wordNum][0]:
                        features['unit_of_size'] = True
                        features2['unit_of_size'] = True
                        features3['unit_of_size'] = True


                # save feature class tuple
                outputFile.write(str((features, classLabel)) + "\n")
                outputFile2.write(str((features2, classLabel)) + "\n")
                outputFile3.write(str((features3, classLabel)) + "\n")
                outputFile4.write(str((features4, classLabel)) + "\n")
                allFeatures.append((features, classLabel))
                allFeatures2.append((features2, classLabel))
                allFeatures3.append((features3, classLabel))
                allFeatures4.append((features4, classLabel))

            # set up feature class tuple
            features5 = {'token':line[wordNum][0].lower()}
            features6 = {'token':line[wordNum][0].lower()}
            features7 = {'token':line[wordNum][0].lower()}
            features8 = {'token':line[wordNum][0].lower()}
            classLabel = line[wordNum][1]

            # add new features
            # stemmed token
            features5['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)
            features7['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)
            features8['stemmed'] = getStem(line[wordNum][0].lower(), stemmer)

            # token shape
            features5['shape'] = getShape(line[wordNum][0])
            features6['shape'] = getShape(line[wordNum][0])
            features7['shape'] = getShape(line[wordNum][0])
            features8['shape'] = getShape(line[wordNum][0])

            # part of speech tag
            features5['pos_tag'] = getPOSTag(line, wordNum)
            features6['pos_tag'] = getPOSTag(line, wordNum)
            features7['pos_tag'] = getPOSTag(line, wordNum)
            features8['pos_tag'] = getPOSTag(line, wordNum)

            # Previous 3 tokens
            # Previous 3 tokens classes
            prev_tokens_classes = getPrevTokensClasses(line, wordNum, stopList)
            features5['prev_tokens'], features5['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()
            features6['prev_tokens'], features6['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()
            features8['prev_tokens'], features8['prev_classes'] = str(prev_tokens_classes[0]).lower(), str(prev_tokens_classes[1]).lower()

            # Next 3 tokens
            features5['next_tokens'] = str(getNextTokens(line, wordNum, stopList))
            features6['next_tokens'] = str(getNextTokens(line, wordNum, stopList))
            features8['next_tokens'] = str(getNextTokens(line, wordNum, stopList))

            # Sub-tokens
            features5['sub-tokens'] = "('','')"
            features6['sub-tokens'] = "('','')"
            features7['sub-tokens'] = "('','')"
            # Contains numbers
            features5['numbers'] = False
            features6['numbers'] = False
            features7['numbers'] = False
            for c in range(len(line[wordNum][0])):
                if line[wordNum][0][c] in numbers:
                    features5['numbers'] = True
                    features6['numbers'] = True
                    features7['numbers'] = True
                    try:
                        if c != 0 and line[wordNum][0][c + 1] not in numbers:
                            features5['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                            features6['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                            features7['sub-tokens'] = "('" + line[wordNum][0].split(line[wordNum][0][c])[0] + line[wordNum][0][c] + "','" + line[wordNum][0].split(line[wordNum][0][c])[1] + "')"
                    except IndexError:
                        pass

            # Contains a unit of size
            features5['unit_of_size'] = False
            features6['unit_of_size'] = False
            features7['unit_of_size'] = False
            for unit in units:
                if unit in line[wordNum][0]:
                    features5['unit_of_size'] = True
                    features6['unit_of_size'] = True
                    features7['unit_of_size'] = True



            # save feature class tuple
            outputFile5.write(str((features5, classLabel)) + "\n")
            outputFile6.write(str((features6, classLabel)) + "\n")
            outputFile7.write(str((features7, classLabel)) + "\n")
            outputFile8.write(str((features8, classLabel)) + "\n")
            allFeatures5.append((features5, classLabel))
            allFeatures6.append((features6, classLabel))
            allFeatures7.append((features7, classLabel))
            allFeatures8.append((features8, classLabel))


    with open(labelled_features_file[:-4] + '.pickle', 'wb+') as f:
        pickle.dump(allFeatures, f)
    with open(labelled_features_file[:-4] + '2.pickle', 'wb+') as f:
        pickle.dump(allFeatures2, f)
    with open(labelled_features_file[:-4] + '3.pickle', 'wb+') as f:
        pickle.dump(allFeatures3, f)
    with open(labelled_features_file[:-4] + '4.pickle', 'wb+') as f:
        pickle.dump(allFeatures4, f)
    with open(labelled_features_file[:-4] + '5.pickle', 'wb+') as f:
        pickle.dump(allFeatures5, f)
    with open(labelled_features_file[:-4] + '6.pickle', 'wb+') as f:
        pickle.dump(allFeatures6, f)
    with open(labelled_features_file[:-4] + '7.pickle', 'wb+') as f:
        pickle.dump(allFeatures7, f)
    with open(labelled_features_file[:-4] + '8.pickle', 'wb+') as f:
        pickle.dump(allFeatures8, f)

    inputFile.close()
    outputFile.close()
    outputFile2.close()
    outputFile3.close()
    outputFile4.close()
    outputFile5.close()
    outputFile6.close()
    outputFile7.close()
    outputFile8.close()
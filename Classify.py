from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer

from nltk.classify import ClassifierI
from statistics import mode
import pickle
from nltk.corpus import stopwords
import string

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        try:
            return mode(votes)
        except Exception as e:
            return votes[0]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        try:
            x = mode(votes)
        except Exception as e:
            x = votes[0]
            
        choice_votes = votes.count(x)
        conf = choice_votes / len(votes)
        return conf

def getStem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)
    
def getShape(word):
    shape = ""

    for char in word:
        if char.isupper():
            shape += 'X'
        else:
            shape += 'x'

    return shape

def getPrevTokens(words, wordNum, stopList):
    prev_tokens = []
    for i in range(1, wordNum + 1):
        if len(prev_tokens) < 3 and words[wordNum - i] not in stopList:
            prev_tokens.append(words[wordNum - i])

    while len(prev_tokens) < 3:
        prev_tokens.append("")

    for i in [0,1,2]:
        prev_tokens[i] = prev_tokens[i].lower()

    return prev_tokens

def getNextTokens(words, wordNum, stopList):
    next_tokens = []
    for i in range(1, len(words) - wordNum):
        if len(next_tokens) < 3 and words[wordNum + i] not in stopList:
            next_tokens.append(words[wordNum + i])

    while len(next_tokens) < 3:
        next_tokens.append("")

    for i in [0,1,2]:
        next_tokens[i] = next_tokens[i].lower()

    return next_tokens

def generateFeaturesNER(sentence):
    allFeatures = []
    stops = stopwords.words('english')
    stops.remove('m')
    stopList = stops + list(string.punctuation)
    stopList.remove("'")
    stopList.remove('"')
    words = word_tokenize(sentence)
    for wordNum in range(len(words)):
        if words[wordNum][0] not in stopList:
            features = {'wordnum':wordNum}

            features['token'] = words[wordNum].lower()
            features['stemmed'] = getStem(words[wordNum].lower())
            features['shape'] = getShape(words[wordNum])
            features['pos_tag'] = pos_tag(words)[wordNum][1]
            features['prev_tokens'] = str(getPrevTokens(words, wordNum, stopList))
            features['next_tokens'] = str(getNextTokens(words, wordNum, stopList))

            allFeatures.append(features)
    return allFeatures

def generateFeaturesRE(sentence, labels):
    sentence = word_tokenize(sentence)
    counter = 1
    ulist = []
    nlist = []
    for label in labels:
        if label[0] == "Bu":
            s = sentence[label[1]]
            try:
                i = 0
                while labels[counter + i][0] ==  "Iu":
                    s += " " + sentence[labels[counter + i][1]]
                    i += 1
            except IndexError:
                pass
            ulist.append((s, counter - 1))
        elif label[0] == "Bn":
            s = sentence[label[1]]
            try:
                i = 0
                while labels[counter + i][0] ==  "In":
                    s += " " + sentence[labels[counter + i][1]]
                    i += 1
            except IndexError:
                pass
            nlist.append((s, counter - 1))

        counter += 1

    allFeatures = []
    for u in ulist:
        for n in nlist:
            features = {}

            # size
            features['size'] = u[0]

            # object
            features['object'] = n[0]

            # size
            if u[1] < n[1]: 
                features['order'] = 'size'
                words_between = sentence[u[1] + 1:n[1]]
            else:
                features['order'] = 'obj'
                words_between = sentence[n[1] + 1:u[1]]

            # num words between
            features['num_words_between'] = str(len(words_between))

            # words between
            features['u_words_between'] = str(list(set(words_between)))

            allFeatures.append(features)

    return allFeatures

def classify(test_name, ent, inputFile, outputFile):
    #load in models
    MNB_F = open("tests/" + test_name + "/MNB_" + ent + ".pickle","rb")
    MNB = pickle.load(MNB_F)
    MNB_F.close()

    BNB_F = open("tests/" + test_name + "/BNB_" + ent + ".pickle","rb")
    BNB = pickle.load(BNB_F)
    BNB_F.close()

    LR_F = open("tests/" + test_name + "/LogisticRegression_" + ent + ".pickle","rb")
    LR = pickle.load(LR_F)
    LR_F.close()

    SGD_F = open("tests/" + test_name + "/SGD_" + ent + ".pickle","rb")
    SGD = pickle.load(SGD_F)
    SGD_F.close()

    SVC_F = open("tests/" + test_name + "/SVC_" + ent + ".pickle","rb")
    SVC = pickle.load(SVC_F)
    SVC_F.close()

    LSVC_F = open("tests/" + test_name + "/LinearSVC_" + ent + ".pickle","rb")
    LSVC = pickle.load(LSVC_F)
    LSVC_F.close()

    # NuSVC_F = open("tests/" + test_name + "/NuSVC_" + ent + ".pickle","rb")
    # NuSVC = pickle.load(NuSVC_F)
    # NuSVC_F.close()

    VC = VoteClassifier(
                        LSVC,
                        SGD,
                        MNB,
                        BNB,
                        LR)

    of = open(outputFile, "w+", encoding="utf-8")
    #open file
    if ent == "ner":
        f = open(inputFile, "r", encoding="utf-8")
        fof = open(outputFile[:-4] + "_features.txt", "w+", encoding="utf-8")
        allClassifictions = []
        allFeaturesClassifictions = []
        lineCount = 0
        for line in f:
            lineCount += 1
            if lineCount % 10 == 0:
                print(lineCount / 1000)
                
            if "<" not in line and "{" not in line and "|" not in line and "===" not in line and "&lt;ref" not in line and "*" not in line and "http" not in line:
                for sentence in sent_tokenize(line):
                    sentence = sentence.replace("[","").replace("]","") 
                    #print("\n#######\n" + sentence)
                    #generate features NER
                    allFeatures = generateFeaturesNER(sentence)

                    classifications = []
                    for featuresNum in range(len(allFeatures)):
                        if featuresNum == 0:
                            allFeatures[featuresNum]["prev_classes"] = "['','','']"
                        elif featuresNum == 1:
                            allFeatures[featuresNum]["prev_classes"] = "['','','" + classifications[featuresNum - 1][0] + "']"
                        elif featuresNum == 2:
                            allFeatures[featuresNum]["prev_classes"] = "['','" + classifications[featuresNum - 2][0] + "','" + classifications[featuresNum - 1][0] + "']"
                        else:
                            allFeatures[featuresNum]["prev_classes"] = "['" + classifications[featuresNum - 1][0] + "','" + classifications[featuresNum - 2][0] + "','" + classifications[featuresNum - 1][0] + "']"

                        place_in_sentence = allFeatures[featuresNum]['wordnum']
                        allFeatures[featuresNum].pop('wordnum')
                        classification = VC.classify(allFeatures[featuresNum])
                        classifications.append((classification, place_in_sentence))
                        #print(allFeatures[featuresNum]["token"], classification, place_in_sentence)
                        allFeaturesClassifictions.append((allFeatures[featuresNum], classification))
                        fof.write(str((allFeatures[featuresNum], classification)) + "\n")
                    
                    bu = False
                    bn = False
                    #print(classifications)
                    for c in classifications:
                        if c[0] == 'Bu':
                            #print("BU TRUE")
                            bu = True
                        elif c[0] == 'Bn':
                            #print("BN TRUE")
                            bn = True

                    if bu and bn:
                        of.write(str((sentence, classifications)) + "\n")
                        allClassifictions.append((sentence, classifications))
                        #add to file

        with open(outputFile[:-4] + '.pickle', 'wb+') as fi:
                pickle.dump(allClassifictions, fi)
                
        with open(outputFile[:-4] + '_features.pickle', 'wb+') as fi:
                pickle.dump(allFeaturesClassifictions, fi)
                
        f.close()

    elif ent == "re":
        inputF = open(inputFile[:-4] + ".pickle","rb")
        inputList = pickle.load(inputF)
        inputF.close()
        classifications = []
        for sentence, labels in inputList:
            of.write(str(sentence) + "\n")
            allFeatures = generateFeaturesRE(sentence, labels)
            for featuresNum in range(len(allFeatures)):
                classification = VC.classify(allFeatures[featuresNum])
                # of.write(str((allFeatures[featuresNum], classification)) + "\n")
                classifications.append((allFeatures[featuresNum], classification))

        with open(outputFile[:-4] + '.pickle', 'wb+') as fi:
                pickle.dump(classifications, fi)

    of.close()
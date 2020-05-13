import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import classification_report

from nltk.classify import ClassifierI
from statistics import mode
import collections


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

def precision(reference, test):
    if len(test) == 0:
        return None
    else:
        return len(reference.intersection(test)) / len(test)



def recall(reference, test):
    if len(reference) == 0:
        return None
    else:
        return len(reference.intersection(test)) / len(reference)


def f1_score(reference, test):
    p = precision(reference, test)
    r = recall(reference, test)

    try:
        return str(2 * ((p * r) / (p + r)))
    except Exception:
        return "None"

def get_model_f1_score(model, testing_set, model_name, logfile):
    y_true = []
    y_pred = []
    for i, (feats, label) in enumerate(testing_set):
        observed = model.classify(feats)
        y_true.append(label)
        y_pred.append(observed)

    print("\n###\n" + model_name)
    print(classification_report(y_true, y_pred))
    logfile.write("\n###\n" + model_name + "\n" + classification_report(y_true, y_pred))
    

def train(labelled_features_file, test_name, ent):
    inputFile = open(labelled_features_file[:-4] + ".pickle","rb")
    featuresets = pickle.load(inputFile)
    inputFile.close()

    # inputFile2 = open(labelled_features_file[:-4] + "_extra.pickle","rb")
    # featuresets2 = pickle.load(inputFile2)
    # inputFile2.close()
    # featuresets += featuresets2

    fair = False
    loops = 0
    if ent == "ner":
        while not fair:
            loops += 1
            random.shuffle(featuresets)
            print("length of featuresets", len(featuresets))
            new_featuresets = featuresets[:int(len(featuresets) * 1)]
            print("NEW length of featuresets", len(new_featuresets))
            training_set = new_featuresets[:int(len(new_featuresets) * 0.9)]
            testing_set =  new_featuresets[int(len(new_featuresets) * 0.9):]

            tro = False
            trbu = False
            triu = False
            trbn = False
            trin = False
            for featureset in training_set:
                if featureset[1] == "o":
                    tro = True
                elif featureset[1] == "Bu":
                    trbu = True
                elif featureset[1] == "Iu":
                    triu = True
                elif featureset[1] == "Bn":
                    trbn = True
                elif featureset[1] == "In":
                    trin = True
            
            teo = False
            tebu = False
            teiu = False
            tebn = False
            tein = False
            for featureset in testing_set:
                if featureset[1] == "o":
                    teo = True
                elif featureset[1] == "Bu":
                    tebu = True
                elif featureset[1] == "Iu":
                    teiu = True
                elif featureset[1] == "Bn":
                    tebn = True
                elif featureset[1] == "In":
                    tein = True

            if tro and trbu and triu and trbn and trin and teo and tebu and teiu and tebn and tein:
                print("FAIR")
                fair = True

            if loops > 500:
                print("LOOOOPS")
                fair = True
                
    else:
        while not fair:
            loops += 1
            random.shuffle(featuresets)
            print("length of featuresets", len(featuresets))
            new_featuresets = featuresets[:int(len(featuresets) * 1)]
            print("NEW length of featuresets", len(new_featuresets))
            training_set = new_featuresets[:int(len(new_featuresets) * 0.9)]
            testing_set =  new_featuresets[int(len(new_featuresets) * 0.9):]

            tro = False
            trr = False
            for featureset in training_set:
                if featureset[1] == "o":
                    tro = True
                elif featureset[1] == "r":
                    trr= True
            
            teo = False
            ter = False
            for featureset in testing_set:
                if featureset[1] == "o":
                    teo = True
                elif featureset[1] == "r":
                    ter = True

            if tro and trr and teo and ter:
                print("FAIR")
                fair = True

            if loops > 100:
                print("LOOOOPS")
                fair = True

    ####################################################
    #  [ (features , class) ]
    #  features = {feature:value, feature:value, ...}
    ####################################################

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    with open("tests/" + test_name + '/MNB_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(MNB_classifier, f)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    with open("tests/" + test_name + '/BNB_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(BernoulliNB_classifier, f)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    with open("tests/" + test_name + '/LogisticRegression_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(LogisticRegression_classifier, f)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    with open("tests/" + test_name + '/SGD_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(SGDClassifier_classifier, f)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    with open("tests/" + test_name + '/SVC_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(SVC_classifier, f)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    with open("tests/" + test_name + '/LinearSVC_' + ent + '.pickle', 'wb+') as f:
        pickle.dump(LinearSVC_classifier, f)

    # NuSVC_classifier = SklearnClassifier(NuSVC(nu=0.2))
    # NuSVC_classifier.train(training_set)
    # with open("tests/" + test_name + '/NuSVC_' + ent + '.pickle', 'wb+') as f:
    #     pickle.dump(NuSVC_classifier, f)

    return testing_set


def test(labelled_features_file, test_name, ent, testing_set):
    # inputFile = open(labelled_features_file[:-4] + ".pickle","rb")
    # featuresets = pickle.load(inputFile)
    # inputFile.close() 
    test_log = open("tests/" + test_name + "/test_log_" + ent + "_" + labelled_features_file[-5:-4] + ".txt", "w+")

    #print(len(testing_set))
            
    #training_set = featuresets[:int(len(featuresets) * 0.9)]
    #testing_set =  featuresets[int(len(featuresets) * 0.9):]

    MNB_F = open("tests/" + test_name + "/MNB_" + ent + ".pickle","rb")
    MNB = pickle.load(MNB_F)
    MNB_F.close()
    get_model_f1_score(MNB, testing_set, "MNB", test_log)

    BNB_F = open("tests/" + test_name + "/BNB_" + ent + ".pickle","rb")
    BNB = pickle.load(BNB_F)
    BNB_F.close()
    get_model_f1_score(BNB, testing_set, "BNB", test_log)

    LR_F = open("tests/" + test_name + "/LogisticRegression_" + ent + ".pickle","rb")
    LR = pickle.load(LR_F)
    LR_F.close()
    get_model_f1_score(LR, testing_set, "LR", test_log)

    SGD_F = open("tests/" + test_name + "/SGD_" + ent + ".pickle","rb")
    SGD = pickle.load(SGD_F)
    SGD_F.close()
    get_model_f1_score(SGD, testing_set, "SGD", test_log)

    SVC_F = open("tests/" + test_name + "/SVC_" + ent + ".pickle","rb")
    SVC = pickle.load(SVC_F)
    SVC_F.close()
    get_model_f1_score(SVC, testing_set, "SVC", test_log)

    LSVC_F = open("tests/" + test_name + "/LinearSVC_" + ent + ".pickle","rb")
    LSVC = pickle.load(LSVC_F)
    LSVC_F.close()
    get_model_f1_score(LSVC, testing_set, "LSVC", test_log)

    # NuSVC_F = open("tests/" + test_name + "/NuSVC_" + ent + ".pickle","rb")
    # NuSVC = pickle.load(NuSVC_F)
    # NuSVC_F.close()
    # get_model_f1_score(NuSVC, testing_set, "NuSVC", test_log)

    VC = VoteClassifier(
                        LSVC,
                        SGD,
                        MNB,
                        BNB,
                        LR)
    get_model_f1_score(VC, testing_set, "VC", test_log)
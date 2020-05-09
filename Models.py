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

    print(classification_report(y_true, y_pred))
    logfile.write(classification_report(y_true, y_pred))
    

def train(labelled_features_file, test_name, ent):
    inputFile = open(labelled_features_file[:-4] + ".pickle","rb")
    featuresets = pickle.load(inputFile)
    inputFile.close()
            
    training_set = featuresets[:int(len(featuresets) * 0.9)]
    #testing_set =  featuresets[int(len(featuresets) * 0.9):]

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

    # NuSVC_classifier = SklearnClassifier(NuSVC())
    # NuSVC_classifier.train(training_set)
    # with open("tests/" + test_name + '/NuSVC_' + ent + '.pickle', 'wb+') as f:
    #     pickle.dump(NuSVC_classifier, f)


def test(labelled_features_file, test_name, ent):
    inputFile = open(labelled_features_file[:-4] + ".pickle","rb")
    featuresets = pickle.load(inputFile)
    inputFile.close()

    test_log = open("tests/" + test_name + "/test_log.txt", "w+")

    print(len(featuresets))
            
    #training_set = featuresets[:int(len(featuresets) * 0.9)]
    testing_set =  featuresets[int(len(featuresets) * 0.9):]

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
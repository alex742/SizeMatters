import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


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

def train(labelled_features_file, test_name):
    inputFile = open(labelled_features_file[:-4] + ".pickle","rb")
    featuresets = pickle.load(inputFile)
    inputFile.close()

    print(len(featuresets))
            
    training_set = featuresets[:int(len(featuresets) * 0.9)]
    testing_set =  featuresets[int(len(featuresets) * 0.9):]

    ####################################################
    #  [ (features , class) ]
    #  features = {feature:value, feature:value, ...}
    ####################################################

    test_log = open("tests/" + test_name + "/test_log.txt", "w+")

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    test_log.write("MNB_classifier accuracy percent: " + (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
    with open('MNB.pickle', 'wb+') as f:
        pickle.dump(MNB_classifier, f)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    test_log.write("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
    with open('BNB.pickle', 'wb+') as f:
        pickle.dump(BernoulliNB_classifier, f)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    test_log.write("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    with open('LogisticRegression.pickle', 'wb+') as f:
        pickle.dump(LogisticRegression_classifier, f)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    test_log.write("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
    with open('SGD.pickle', 'wb+') as f:
        pickle.dump(SGDClassifier_classifier, f)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    test_log.write("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
    with open('SVC.pickle', 'wb+') as f:
        pickle.dump(SVC_classifier, f)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    test_log.write("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
    with open('LinearSVC.pickle', 'wb+') as f:
        pickle.dump(LinearSVC_classifier, f)

    # NuSVC_classifier = SklearnClassifier(NuSVC())
    # NuSVC_classifier.train(training_set)
    # test_log.write("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
    # with open('NuSVC.pickle', 'wb+') as f:
    #     pickle.dump(NuSVC_classifier, f)

    voted_classifier = VoteClassifier(
                                    LinearSVC_classifier,
                                    SGDClassifier_classifier,
                                    MNB_classifier,
                                    BernoulliNB_classifier,
                                    LogisticRegression_classifier)

    test_log.write("vote_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



def test(test_name):
    print(test_name)
    MNB_F = open("tests/" + "test_name" + "/MNB.pickle","rb")
    MNB = pickle.load(MNB_F)
    MNB_F.close()
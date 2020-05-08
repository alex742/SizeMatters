from CandidateSentences import getCandidateSentences
from LabelledSentences import getLabelledSentences
from LabelledFeatures import getLabelledFeatures
from Models import train, test

import time
import os

test_name = input("Name of test: ")
test_name = test_name + str(time.time()).split(".")[0]
os.mkdir("tests/" + test_name)

input_text_file = "data/test.txt"
candidate_sentences_file = "tests/" + test_name + "/" + "candidate_sentences.txt"
labelled_sentences_file = "tests/" + test_name + "/" + "labelled_sentences.txt"
labelled_features_file = "tests/" + test_name + "/" + "labelled_features.txt"

getCandidateSentences(input_text_file, candidate_sentences_file)
getLabelledSentences(candidate_sentences_file, labelled_sentences_file)
getLabelledFeatures(labelled_sentences_file, labelled_features_file)
# train(labelled_features_file, test_name)
# test(labelled_features_file, test_name)
# train(labelled_features_file[:-4] + "_relationships.txt", test_name)
# test(labelled_features_file[:-4] + "_relationships.txt", test_name)
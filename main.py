from CandidateSentences import getCandidateSentences
from LabelledSentences import getLabelledSentences
from LabelledFeatures import getLabelledFeatures
from Models import train, test
from Classify import classify

import time
import os

test_name = input("Name of test: ")
test_name = test_name + str(time.time()).split(".")[0]
os.mkdir("tests/" + test_name)

#test_name = "gold1588984696"
test_name = "all_features_preprocessing_on_gold"
#test_name = "all_features_noprocessing_on_gold"

input_text_file = "data/enwiki-latest-pages-articles1.xml-p1p30303"
#input_text_file = "data/gold.txt"
#input_text_file = "data/test.txt"

# candidate_sentences_file = "tests/" + test_name + "/" + "candidate_sentences.txt"
labelled_sentences_file = "tests/" + test_name + "/" + "labelled_sentences.txt"
labelled_features_file = "tests/" + test_name + "/" + "labelled_features.txt"

# getCandidateSentences(input_text_file, candidate_sentences_file)
# getLabelledSentences(candidate_sentences_file, labelled_sentences_file)
# getLabelledFeatures(labelled_sentences_file, labelled_features_file)

# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + ".txt", test_name, "ner")
# test(labelled_features_file[:-4] + ".txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "2.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "2.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "3.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "3.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "4.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "4.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "5.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "5.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "6.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "6.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "7.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "7.txt", test_name, "ner", testing_set)
# print("\n\n\n####################################\nNER")
# testing_set = train(labelled_features_file[:-4] + "8.txt", test_name, "ner")
# test(labelled_features_file[:-4] + "8.txt", test_name, "ner", testing_set)

# print("\n\n\n####################################\nRE")
# testing_set = train(labelled_features_file[:-4] + "_relationships.txt", test_name, "re")
# test(labelled_features_file[:-4] + "_relationships.txt", test_name, "re", testing_set)

classify(test_name, "ner", input_text_file, "tests/" + test_name + "/testOutputNER.txt")
classify(test_name, "re", "tests/" + test_name + "/testOutputNER.txt", "tests/" + test_name + "/testOutputRE.txt")
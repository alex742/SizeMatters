from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import re

text = open("trainingText.txt", "r")
candidates = open("candidateSentences.txt", "w+")
candidatesUnits = open("candidateSentencesUnits.txt", "w+")

units = '[0-9][0-9]* ?([a-z][a-z]?|(kilo|centi|milli)?meters?|(kilo|centi|milli)?metres?|inch(es)?|f(oo|ee)t|"|\'|yards?|miles?)( |\.|!|\?)'
unitsRE = re.compile(units)

"""
Iterate through sentences in text file
For each sentence:
	Is there a noun? - WordNet objects
	Is there a noun? - nltk pos tagger
	Is there a size? - regex

	if yes:
		add sentence to candidates
	else:
		continue

"""

for line in text:
    for sentence in sent_tokenize(line):
        if unitsRE.search(sentence):
            candidates.write(sentence + "\n")
            candidatesUnits.write(sentence + "\n")
        else:
            tagged_sentence = pos_tag(word_tokenize(sentence))
            for word, tag in tagged_sentence:
                # NN (Noun), NNS (Nouns), NNP (Proper Noun), NNPS (Proper Nouns)
                if 'NN' in tag:
                    candidates.write(sentence + "\n")
                    break

text.close()
candidates.close()
candidatesUnits.close()
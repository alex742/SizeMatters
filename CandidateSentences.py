from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import re

def getCandidateSentences(input_file, output_file):
    text = open(input_file, "r", encoding="utf-8")
    candidates = open(output_file, "w+", encoding="utf-8")
    #candidatesUnits = open("candidateSentencesUnits.txt", "w+", encoding="utf-8")

    #units = '[0-9][0-9]* ?([a-z][a-z]?|(kilo|centi|milli)?meters?|(kilo|centi|milli)?metres?|inch(es)?|f(oo|ee)t|"|\'|yards?|miles?)( |\.|!|\?)'
    units = '[0-9][0-9]* ?([a-m]?[o-q]?[u-z]?[a-z]?|(kilo|centi|milli)?meters?|(kilo|centi|milli)?metres?|inch(es)?|f(oo|ee)t|"|\'|yards?|miles?)( |\.|!|\?)'
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
    line_count = 0
    for line in text:
        line_count += 1
        for sentence in sent_tokenize(line):
            if unitsRE.search(sentence) and "<" not in sentence and "{" not in sentence and "|" not in sentence and "===" not in sentence:
                candidates.write(sentence + "\n")
                #candidatesUnits.write(sentence + "\n")
            else:
                pass
                tagged_sentence = pos_tag(word_tokenize(sentence))
                for word, tag in tagged_sentence:
                    # NN (Noun), NNS (Nouns), NNP (Proper Noun), NNPS (Proper Nouns)
                    if 'NN' in tag:
                        candidates.write(sentence + "\n")
                        break
        if line_count % 10000 == 0:
            print(line_count / 10000)


    text.close()
    candidates.close()
    #candidatesUnits.close()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

inputFile = open("candidateSentencesUnits.txt", "r")
outputFile = open("labelledSentences.txt", "a+")
wordsLabels = []

for line in inputFile:
    words = word_tokenize(line)
    print(line)

    label = ""
    if label == "q":
        for word in words:
            


inputFile.close()
outputFile.close()
from nltk.tokenize import word_tokenize
import pickle

inputFile = open("candidateSentencesUnits.txt", "r")
outputFile = open("labelledSentences.txt", "a+")

sentenceLabels = []
for line in inputFile:
    words = word_tokenize(line)
    wordLabels = [] #line[:-2]

    label = ""
    for word in words:
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" + line)
        
        # if user input 'q' then until end of line keep input as 'q'
        label = "" if label != 'q' else 'q'

        # loop getting input until valid input provided
        while label != 'o' and label != 'q' and label != 'u' and label != 'n':
            label = input(word + " ")

        # try to check previous label to see if B or I
        if label == 'n' or label == 'u':
            try:
                if label in wordLabels[-1][1]:
                    label = 'I' + label
                else:
                    label = 'B' + label
            except IndexError:
                label = 'B' + label

            wordLabels.append((word, label))
        
        else:
            # works for both 'q' and 'o'
            wordLabels.append((word, 'o'))

    outputFile.write(str(wordLabels) + '\n')
    sentenceLabels.append(wordLabels)

with open('labelledSentences.pickle', 'wb+') as f:
    pickle.dump(sentenceLabels, f)

inputFile.close()
outputFile.close()
from nltk.tokenize import word_tokenize
import pickle

def getLabelledSentences(candidate_sentences_file, labelled_sentences_file):
    inputFile = open(candidate_sentences_file, "r", encoding="utf-8")
    outputFile = open(labelled_sentences_file, "w+", encoding="utf-8")
    routputFile = open(labelled_sentences_file[:-4].replace("sentences","features") + "_relationships.txt", "w+", encoding="utf-8")

    sentenceLabels = []
    rLabels = []
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

        counter = 1
        u = 0
        lu = 0
        n = 0
        ln = 0
        words_between = []
        for wordLabel in wordLabels:
            if wordLabel[1] == "Bu" and u == 0:
                u = counter
                i = 0
                try:
                    while wordLabels[counter + i][1] ==  "Iu":
                        i += 1
                except IndexError:
                    pass
                lu = i
            elif wordLabel[1] == "Bn" and n == 0:
                n = counter
                i = 0
                try:
                    while wordLabels[counter + i][1] ==  "In":
                        i += 1
                except IndexError:
                    pass
                ln = i

            if u != n and (n == 0 or u == 0):
                words_between.append(wordLabel[0])
            counter += 1

        if u and n:
            # label as relationship
            print("\nRelationship?\n\n" + line)
            r = input("")

            features = {}

            # size
            us = str(wordLabels[u-1][0])
            for i in range(lu):
                us += " " + str(wordLabels[u + i][0])
            features['size'] = us

            # object
            ns = str(wordLabels[n-1][0])
            for i in range(ln):
                ns += " " + str(wordLabels[n + i][0])
            features['object'] = ns

            # size
            if u < n: 
                features['order'] = 'size'
                words_between = words_between[lu + 1:]
            else:
                features['order'] = 'obj'
                words_between = words_between[ln + 1:]

 
            # num words between
            features['num_words_between'] = str(len(words_between))

            # words between
            features['u_words_between'] = str(list(set(words_between)))

            if r == "r":
                routputFile.write(str((features, 'r')) + "\n")
                rLabels.append((features, 'r'))
            else:
                routputFile.write(str((features, 'o')) + "\n")
                rLabels.append((features, 'o'))
            



    with open(labelled_sentences_file[:-4] + '.pickle', 'wb+') as f:
        pickle.dump(sentenceLabels, f)

    with open(labelled_sentences_file[:-4].replace("sentences","features") + '_relationships.pickle', 'wb+') as f:
        pickle.dump(rLabels, f)

    inputFile.close()
    outputFile.close()
    routputFile.close()
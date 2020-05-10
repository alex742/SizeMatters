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
            if label != 'q':
                print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" + line)
            
            # if user input 'q' then until end of line keep input as 'q'
            if label == 'q':
                label = 'q'
            elif label == "stop":
                label = "stop"
            else:
                label = ''

            # loop getting input until valid input provided
            while label != 'o' and label != 'q' and label != 'u' and label != 'n' and label != "stop":
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
        ulist = []
        nlist = []
        for wordLabel in wordLabels:
            if wordLabel[1] == "Bu":
                s = wordLabel[0]
                try:
                    i = 0
                    while wordLabels[counter + i][1] ==  "Iu":
                        s += " " + wordLabels[counter + i][0]
                        i += 1
                except IndexError:
                    pass
                ulist.append((s, counter - 1))
            elif wordLabel[1] == "Bn":
                s = wordLabel[0]
                try:
                    i = 0
                    while wordLabels[counter + i][1] ==  "In":
                        s += " " + wordLabels[counter + i][0]
                        i += 1
                except IndexError:
                    pass
                nlist.append((s, counter - 1))

            counter += 1

        for u in ulist:
            for n in nlist:
                # label as relationship
                print("\nRelationship?\n\n" + line + "\n" + str(u) + " --- " + str(n) + "\n")
                r = input("Label: ")

                features = {}

                # size
                features['size'] = u[0]

                # object
                features['object'] = n[0]

                # size
                if u[1] < n[1]: 
                    features['order'] = 'size'
                    words_between = wordLabels[u[1] + 1:n[1]]
                else:
                    features['order'] = 'obj'
                    words_between = wordLabels[n[1] + 1:u[1]]

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
                

        if label == "stop":
            with open(labelled_sentences_file[:-4] + '.pickle', 'wb+') as f:
                pickle.dump(sentenceLabels, f)

            with open(labelled_sentences_file[:-4].replace("sentences","features") + '_relationships.pickle', 'wb+') as f:
                pickle.dump(rLabels, f)

            inputFile.close()
            outputFile.close()
            routputFile.close()
            quit()

    with open(labelled_sentences_file[:-4] + '.pickle', 'wb+') as f:
        pickle.dump(sentenceLabels, f)

    with open(labelled_sentences_file[:-4].replace("sentences","features") + '_relationships.pickle', 'wb+') as f:
        pickle.dump(rLabels, f)

    inputFile.close()
    outputFile.close()
    routputFile.close()
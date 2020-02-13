from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

f = open("trainingText.txt", "r")

wordsLabels = []

for line in f:
    sents = sent_tokenize(line)
    for sent in sents:
        words = word_tokenize(sent)
        
        print(sent)
        q = False
        for word in words:
            if not q:
                if word not in stop_words:
                    print(word, end="")
                    label = input("   ")
                if label != "q":
                    wordsLabels.append((word, label))
                else:
                    q = True
                    wordsLabels.append((word, "n"))
            else:
                wordsLabels.append((word, "n"))

f.close()

f = open("trainingData.txt", "a+")
for wordLabel in wordsLabels:
    f.write(str(wordLabel) + "\n")
f.close()
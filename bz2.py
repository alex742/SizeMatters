o = open("wiki_abstracts.txt", "w+", encoding='utf-8')
f = open("enwiki-latest-abstract.xml", "r", encoding='utf-8')

s = ""
for line in f:
    try:
        #if "<url>" in line:
            #s += line.replace("<url>","").replace("</url>","")
        if "<abstract>" in line:
            o.write(s + line.replace("<abstract>","").replace("</abstract>",""))
            s = ""
    except UnicodeDecodeError:
        pass
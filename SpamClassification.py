import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

stopwords=set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
df=pd.read_csv('hamspam2.csv')

spam=[]
ham=[]
mails=[]

for i in range(len(df.iloc[:,1])):
    s=df.iloc[i,0]
    if(df.iloc[i,1]=='Ham'):
        ham.append((s,'ham'))
    else:
        spam.append((s,'spam'))


for (w,k) in spam + ham:
    words=word_tokenize(w)
    wfilt=[wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]
    wfilt=[w.lower() for w in wfilt if len(w)>=3]
    mails.append((wfilt,k))

def get_words(mails):
    all=[]
    for (words, know) in mails:
        all.extend(words)
    return (all)

def get_feat(wordlist):
    wordlist=nltk.FreqDist(wordlist)
    wfeat=wordlist.keys()
    return wfeat

wfeat=get_feat(get_words(mails))

wfeat_filt=[]
for w in wfeat:
    if w not in stopwords:
        wfeat_filt.append(w)

def extract(d):
    doc=set(d)
    feat={}
    for word in wfeat_filt:
        feat['contains(%s)' %word] = (word in doc)
    return feat

training_set = nltk.classify.apply_features(extract, mails)
classifier = nltk.NaiveBayesClassifier.train(training_set)

testmail='Join ICICI bank and get amazing offers'
print("{}: Classified as {}".format(testmail, classifier.classify(extract(testmail.split()))))

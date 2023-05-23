import tkinter
from textblob import TextBlob
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from tkinter import filedialog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

main = tkinter.Tk()
main.title("Analysis of Women Safety in Indian Cities Using Twitter data")
main.geometry("1200x1200")

global filename
tweets_list = []
clean_list = []
global pos, neu, neg, pos_vader, neu_vader, neg_vader

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");

def read():
    tweets_list.clear()
    train = pd.read_csv(filename, encoding='iso-8859-1')
    for i in range(len(train)):
        tweet = train._get_value(i, 'Text')
        tweets_list.append(tweet)
        text.insert(END, tweet + "\n")

def clean():
    text.delete('1.0', END)
    clean_list.clear()
    for i in range(len(tweets_list)):
        tweet = tweets_list[i]
        tweet = tweet.strip("\n")
        tweet = tweet.strip()
        tweet = tweetCleaning(tweet.lower())
        clean_list.append(tweet)
        text.insert(END, tweet + "\n")

def machineLearning():
    text.delete('1.0', END)
    global pos, neu, neg
    pos = 0
    neu = 0
    neg = 0
    for tweet in clean_list:
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            pos += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : POSITIVE\n")
            text.insert(END, "Polarity Score      : " + str(blob.polarity) + "\n")
            text.insert(END, '====================================================================================\n')
        elif polarity < 0:
            neg += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : NEGATIVE\n")
            text.insert(END, "Polarity Score      : " + str(blob.polarity) + "\n")
            text.insert(END, '====================================================================================\n')
        else:
            neu += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : NEUTRAL\n")
            text.insert(END, "Polarity Score      : " + str(blob.polarity) + "\n")
            text.insert(END, '====================================================================================\n')

def vaderSentimentAnalysis():
    text.delete('1.0', END)
    global pos_vader, neu_vader, neg_vader
    pos_vader = 0
    neu_vader = 0
    neg_vader = 0
    analyzer = SentimentIntensityAnalyzer()
    for tweet in clean_list:
        scores = analyzer.polarity_scores(tweet)
        if scores['compound'] <= -0.05:  # Negative
            neg_vader += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : NEGATIVE\n")
            text.insert(END, "Compound Score      : " + str(scores['compound']) + "\n")
            text.insert(END, '====================================================================================\n')
        elif scores['compound'] >= 0.05:  # Positive
            pos_vader += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : POSITIVE\n")
            text.insert(END, "Compound Score      : " + str(scores['compound']) + "\n")
            text.insert(END, '====================================================================================\n')
        else:  # Neutral
            neu_vader += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment : NEUTRAL\n")
            text.insert(END, "Compound Score      : " + str(scores['compound']) + "\n")
            text.insert(END, '====================================================================================\n')

def graph():
    labels = ['Positive', 'Negative', 'Neutral']
    sizes_textblob = [pos, neg, neu]
    sizes_vader = [pos_vader, neg_vader, neu_vader]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.pie(sizes_textblob, labels=labels, autopct='%1.1f%%', startangle=140)
    ax1.set_title('TextBlob Sentiment Analysis')

    ax2.pie(sizes_vader, labels=labels, autopct='%1.1f%%', startangle=140)
    ax2.set_title('VADER Sentiment Analysis')

    plt.axis('equal')
    plt.show()

def tweetCleaning(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

font = ('times', 16, 'bold')
title = Label(main, text='Analysis of Women Safety in Indian Cities Using Twitter data')
title.config(bg='black', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Tweets Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

readButton = Button(main, text="Read Tweets", command=read)
readButton.place(x=200, y=100)
readButton.config(font=font1)

cleanButton = Button(main, text="Clean Tweets", command=clean)
cleanButton.place(x=350, y=100)
cleanButton.config(font=font1)

mlButton = Button(main, text="TextBlob Sentiment Analysis", command=machineLearning)
mlButton.place(x=500, y=100)
mlButton.config(font=font1)

vaderButton = Button(main, text="VADER Sentiment Analysis", command=vaderSentimentAnalysis)
vaderButton.place(x=750, y=100)
vaderButton.config(font=font1)

graphButton = Button(main, text="Show Analysis Graph", command=graph)
graphButton.place(x=1000, y=100)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)
text.config(font=font1)

main.config(bg='coral')
main.mainloop()

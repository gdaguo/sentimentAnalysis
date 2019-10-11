import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


if __name__=='__main__':
  #load positive and negative reviews
  positive_fileids = movie_reviews.fileids('pos')
  negative_fileids = movie_reviews.fileids('neg')

features_positive = [(extract_features(movie_reviews.words(fileids=[f])),'Hit Movie/Positive ') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),'Flop Movie/Negative') for f in negative_fileids]

#split the data into train and test (80/20)
threshold_factor = 0.7
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

#feature train and test model
features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
print(("\nNumber of training datapoints:"), len(features_train))
print(("Number of test datapoints:"), len(features_test))

#train a naive bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print(("\nAccuracy of the classifier:"), nltk.classify.util.accuracy(classifier, features_test))
print ("\nTop 15 most informative words:")
for item in classifier.most_informative_features()[:15]:
    print (item[0])

#tkinter GUI
root = Tk()

root.geometry('200x150')

input_reviews = askopenfile(mode ='r', filetypes =[('file', '*.txt')])
#btn = Button(root, command = lambda:open_file()) 
#Button.pack(self, side = TOP, pady = 10)
#button = Button(root, text="Close Me", command=root.destroy)
#button.pack()
root.destroy()

root.mainloop()

#Prediction and Results

print ("\nPrediction:")
for review in input_reviews:
    print(("\nReview:"), review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print(("Predicted sentiment:"), pred_sentiment)
    print(("Probability:"), round(probdist.prob(pred_sentiment),2));
    












import flask
import numpy as np
import pandas as pd
import pickle
import itertools
import json
import seaborn as sns
import math
import nltk, string
import re
import random
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from time import time
from scipy import sparse
from gensim.summarization import summarize
from datetime import datetime

with open('tfidf_vector.pkl', 'rb') as picklefile:
	tfidf = pickle.load(picklefile)
with open('question_list.pkl', 'rb') as picklefile:
	question_list = pickle.load(picklefile)
with open('title_list.pkl', 'rb') as picklefile:
	title_list = pickle.load(picklefile)
with open('answer_list.pkl', 'rb') as picklefile:
	answer_list = pickle.load(picklefile) 
with open('score_list.pkl', 'rb') as picklefile:
	score_list = pickle.load(picklefile)
with open('flair_list.pkl', 'rb') as picklefile:
	flair_list = pickle.load(picklefile)

# This is the part that takes 20 seconds
print(str(datetime.now()) + ' starting tfidf.transform')
corpus = tfidf.transform(question_list)
print(str(datetime.now()) + ' tfidf.transform complete')

# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open("relationship-app.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/relationships", methods=["POST"])
def answer():

	print(str(datetime.now()) + ' starting analysis')
	data = flask.request.json
	test_case_list = data["question"]
	test_case = test_case_list[0]
	# stop_words = stopwords.words("english")
	stop_words = set(nltk.corpus.stopwords.words('english'))
	# stop_words.add(('lady', 'son','ask', 'her', 'she', 'girlfriend', 'west coast',
	# 	'east coast'))

	# take my "stop words" out of the user question
	ignore_words = ['lady', 'son','ask', 'her', 'she', 'girlfriend', 'coast', 'coast.']
	test_case_words = test_case.lower().split()
	result_words = [word for word in test_case_words if word.lower() not in ignore_words]
	test_case = " ".join(result_words)
	print test_case
	
	test_case_list = [test_case]

	test_case_vector = tfidf.transform(test_case_list)
	distances = pd.DataFrame(cosine_similarity(corpus, test_case_vector))

	# Append the real questions, answers, etc to the distances frame
	distances['title'] = title_list
	distances['question'] = question_list
	distances['answer'] = answer_list
	distances['score'] = score_list
	distances['flair'] = flair_list

	# Remove deleted answers 
	distances = distances.loc[distances.answer != "[deleted]", ]

	# Rename
	distances = distances.rename(columns = {0:"similarity"})

	hundred_largest = distances.nlargest(500, "similarity")

	def run_topic_model(qlist, test_case, stop_words, n_topics):
		"""Run Topic Model on List of Questions; Apply Model on Test Question"""
		vectorizer = TfidfVectorizer(stop_words = stop_words)
		tfidf = vectorizer.fit_transform(qlist)
		# Fit the NMF topic modeling on the matrix
		topics = NMF(n_components = n_topics, random_state = 2,
			alpha = .1, l1_ratio = .5).fit(tfidf)
		# Fit the vectorizer on the question_list
		qlist_2 = vectorizer.fit_transform(qlist)
		qmatrix = topics.transform(qlist_2)
		# Fit everything on the test_case question
		tlist = []
		tlist.append(test_case)
		tlist_2 = vectorizer.transform(tlist)
		tmatrix = topics.transform(tlist_2)
		return qmatrix, tmatrix

	# Reset index on data frame and run function
	# n_topics = 20
	n_topics = 20
	hundred_largest = hundred_largest.reset_index()
	
	qmatrix, tmatrix = run_topic_model(hundred_largest['question'], test_case, stop_words, n_topics)

	distances2 = pd.DataFrame(cosine_similarity(qmatrix, tmatrix))

	# Append the real questions, answers, etc to the distances frame
	distances3 = pd.concat([hundred_largest, distances2], axis = 1)
	distances3 = distances3.sort_values(0, ascending = False).head(10)

	# Rename topic similarity
	distances3 = distances3.rename(columns = {0:"topic similarity"})

	# Sort by similarity
	distances3.sort_values(['similarity'], inplace=True, ascending=False)
	distances3 = distances3.head()
	a = ' '.join(distances3['answer'])

	## GENSIM Summarizer
	# the_answer = summarize(a, split = True)[0]
	
	# Subjectivity
	list_of_sent = list(distances3['answer'])
	sentiment_list = []
	for x in list_of_sent:
		blob = TextBlob(x)
		sentiment_list.append(blob.sentiment.subjectivity)
	top_answers = []
	# for c in sorted(sentiment_list, reverse = True)[:3]:
	for c in sorted(sentiment_list, reverse = True)[:4]:
		top_answers.append(list_of_sent[sentiment_list.index(c)])
	the_result = ' '.join(top_answers)

	the_answer = summarize(the_result, split = True)[0]

	print(str(datetime.now()) + ' analysis complete')

	# Make answer gender specific just in case
	if "my boyfriend" in test_case.lower():
		the_answer = the_answer.replace(" her ", " him ")
		the_answer = the_answer.replace(" her,", " him,")
		the_answer = the_answer.replace("she", "he")

	if "my husband" in test_case.lower():
		the_answer = the_answer.replace(" her ", " him ")
		the_answer = the_answer.replace("she", "he")

	if "my girlfriend" in test_case.lower():
		the_answer = the_answer.replace("him", "her")
		the_answer = the_answer.replace("his", "her")
		the_answer = the_answer.replace(" he ", " she ")
		the_answer = the_answer.replace("boyfriend", "girlfriend")


	return flask.jsonify({'answer':the_answer})

app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

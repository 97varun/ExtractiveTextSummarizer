from nltk.tokenize import sent_tokenize, RegexpTokenizer
from collections import Counter
import math
import pickle

NUM_INS = 4396
NUM_CNN = 10000

class ExtractiveSummarizer :
	def __init__(self, corpus = "cnn", scoring = 'tf-idf') :
		if scoring != "tf-idf" and scoring != "bayes" :
			print("error: scoring scheme unavailable")
			return
		if corpus != "cnn" and corpus != "ins" :
			print("error: corpus unavailable")
			return

		num_sample = NUM_CNN if corpus == "cnn" else NUM_INS
		with open(".\\{}\\{}{}.pkl".format(corpus, corpus, num_sample), "rb") as fp :
			self.corpus = pickle.load(fp)

		if scoring == 'tf-idf' :
			self.init_idf()
		elif scoring == 'bayes' :
			self.init_word_score()
		
		self.scoring = scoring

	# TF-IDF FUNCTIONS

	def init_idf(self) :
		# number of documents
		N = len(self.corpus[0])

		# document frequency - number of documents a particular word occurs in.
		df = {}
		for doc in self.corpus[0] :
			for word in set(doc) :
				df[word] = df.get(word, 0) + 1

		self.idf = {x : math.log(N / df[x], 10) for x in df}

	def get_idf(self, word) :
		N = len(self.corpus[0])
		# unknown words have idf value = log10(N), can try 0 also
		return self.idf.get(word, math.log(N, 10))

	# BAYES FUNCTIONS

	def init_word_score(self) :
		article_word_list = [word for doc in self.corpus[0] for word in doc]
		summary_word_list = [word for doc in self.corpus[1] for word in doc]

		Na = len(article_word_list) # total number of words in the articles
		Ns = len(summary_word_list) # total number of words in the summaries

		article_word_dict = Counter(article_word_list)
		summary_word_dict = Counter(summary_word_list)
		combined_dict = article_word_dict + summary_word_dict

		Va = len(article_word_dict) # number of unique words in the articles
		Vs = len(summary_word_dict) # number of unique words in the summaries

		# P(Wi | Sw)
		P_Wi_given_Sw = Counter({word : summary_word_dict[word] / Ns for word in summary_word_dict})

		# P(Sw)
		P_Sw = Ns / (Na + Ns)

		# P(Wi)
		P_Wi = Counter({word : (combined_dict[word]) / (Na + Ns) for word in combined_dict})

		# P(Sw | Wi)
		self.word_score = Counter({word : P_Wi_given_Sw[word] * P_Sw / P_Wi[word] for word in article_word_dict})

		# print("Va: ", Va)
		# print("Vs: ", Vs)
		# print("Na: ", Na)
		# print("Ns: ", Ns)

	# SUMMARISER FUNCTIONS

	def get_sorted_indices(self, sentences) :
		indices = list(range(len(sentences)))

		if self.scoring == 'tf-idf' :
			# term frequency - number of times a word occurs in a document.
			tf = {}
			for sentence in sentences :
				for word in sentence :
					if word == '.' or word == ',' or word == '?' or word == '"' or word == '\'' or word == '(' or word == ')' or word == '“' or word == '”' :
						continue
					tf[word] = tf.get(word, 0) + 1
			indices.sort(key = lambda i : sum([tf.get(word, 0) * self.get_idf(word) for word in sentences[i]]) / len(sentences[i]), reverse = True)

		elif self.scoring == 'bayes' :
			indices.sort(key = lambda i : sum([self.word_score.get(word, 0) for word in sentences[i]]) / len(sentences[i]), reverse = True)

		return indices

	def summarizer(self, text, sentence_count = 5) :
		if sentence_count <= 0 :
			print("error: sentence_count has to be a positive integer!")
			return
		
		# preprocess
		rem_list = ['“', '”', '"']
		for rem in rem_list :
			text = text.replace(rem, '')
		
		sentences = sent_tokenize(text)
		original_sentences = list(sentences)

		mytokenizer = RegexpTokenizer(r'\d+\.\d+|[^\W\d]+|\d+')
		sentences = [mytokenizer.tokenize(sentence) for sentence in sentences]

		sentences = [[word.lower() for word in sentence] for sentence in sentences]

		# print(sentences)

		# number of sentences
		N = len(sentences)

		if sentence_count > N :
			print("error: sentence_count greater that number of sentences in the article!")
			return

		# sort sentence indices based on tf-idf score
		indices = list(range(N))

		indices = self.get_sorted_indices(sentences)
		indices = set(indices[:sentence_count])

		# append highest scoring sentences in order to get the summary
		summary = []
		for i in range(len(original_sentences)) :
			if i in indices :
				summary.append(original_sentences[i])
		summary = " ".join(summary)
		return summary

if __name__ == "__main__" :
	text = """President Trump lashed out Tuesday at the publication of questions that special counsel Robert S. Mueller III was said to be interested in asking him as part of the Russia probe and possible attempts to obstruct the inquiry.
In a morning tweet, Trump said it was “disgraceful” that the 49 questions were provided to the New York Times, which published them Monday night.
“So disgraceful that the questions concerning the Russian Witch Hunt were ‘leaked’ to the media,” he wrote on Twitter.
It appears that the leak did not come from Mueller’s office. The Times reported that the questions were relayed to Trump’s attorneys as part of negotiations over the terms of a potential interview with the president. The list was then provided to the Times by a person outside Trump’s legal team, the paper said.
In his tweet, Trump also falsely asserts that there are no questions about “Collusion.” Among those is a query about Trump’s knowledge of any outreach by his former campaign chairman Paul Manafort to Russia “about potential assistance to the campaign.” A court filing this month revealed that Mueller had sought authorization to expand his probe into allegations that Manafort “committed a crime or crimes by colluding with Russian government officials.”
Another question asks about Trump’s knowledge of a June 2016 meeting in Trump Tower between his aides and a Russian lawyer who offered politically damaging information on Trump’s Democratic opponent, Hillary Clinton.
And another asks what Trump knew about “Russian hacking, use of social media or other acts aimed at the campaign?”
In his tweet, Trump calls collusion “a phony crime” and repeats his claim that none existed. The president also derides Mueller’s investigation as having “begun with illegally leaked classified information,” adding: “Nice!”"""

	print("SUMMARY USING CNN CORPUS AND TF-IDF SCORING", end = "\n\n")
	text_summarizer1 = ExtractiveSummarizer()
	summary = text_summarizer1.summarizer(text, sentence_count = 3)
	print(summary, end = "\n\n")

	print("SUMMARY USING CNN CORPUS AND BAYES SCORING", end = "\n\n")
	text_summarizer2 = ExtractiveSummarizer(scoring = "bayes")
	summary = text_summarizer2.summarizer(text, sentence_count = 3)
	print(summary, end = "\n\n")

	print("SUMMARY USING INSHORTS CORPUS AND TF-IDF SCORING", end = "\n\n")
	text_summarizer3 = ExtractiveSummarizer(corpus = "ins")
	summary = text_summarizer3.summarizer(text, sentence_count = 3)
	print(summary, end = "\n\n")

	print("SUMMARY USING INSHORTS CORPUS AND BAYES SCORING", end = "\n\n")
	text_summarizer4 = ExtractiveSummarizer(corpus = "ins", scoring = "bayes")
	summary = text_summarizer4.summarizer(text, sentence_count = 3)
	print(summary, end = "\n\n")
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

def pre_cnn() :
	with open(".\\cnn\\cnn_dataset.pkl", "rb") as fp :
		cnn = pickle.load(fp)
	stories = []
	highlights = []

	NUM_STORIES = 10000
	for i in cnn[:NUM_STORIES] :
		story = i['story']
		highlight = i["highlights"]
		stories.append(" ".join(story))
		highlights.append(" ".join(highlight))

	gen_data("cnn".format(NUM_STORIES), stories, highlights)
	# gen_word_list()

def pre_ins() :
	with open('.\\ins\\ins_dataset.pkl', 'rb') as fp :
		a, s = pickle.load(fp)
	gen_data("ins", a, s)

def gen_data(corpus, x_master, y_master) :
	stop_words = set(stopwords.words('english'))

	mytokenizer = RegexpTokenizer(r'\d+\.\d+|[^\W\d]+|\d+')

	article_list = []
	summary_list = []

	for article_x in x_master :
	    words = list(filter(lambda x : x not in stop_words, map(lambda y : y.lower(), mytokenizer.tokenize(str(article_x)))))
	    article_list.append(words)

	for summary_y in y_master :
	    words = list(filter(lambda x : x not in stop_words, map(lambda y : y.lower(), mytokenizer.tokenize(str(summary_y)))))
	    summary_list.append(words)

	with open(".\\{}\\{}{}.pkl".format(corpus, corpus, len(x_master)), 'wb') as fp :
	    pickle.dump((article_list, summary_list), fp)

if __name__ == "__main__"  :
	# pre_cnn()
	# pre_ins()
	
	with open(".\\cnn\\cnn10000.pkl", "rb") as fp :
		cnn = pickle.load(fp)
	print(len(cnn))
	print(len(cnn[0]), len(cnn[1]))
	print(len(cnn[0][0]), len(cnn[1][0]))

	with open(".\\ins\\ins4396.pkl", "rb") as fp :
		ins = pickle.load(fp)
	print(len(ins))
	print(len(ins[0]), len(ins[1]))
	print(len(ins[0][0]), len(ins[1][0]))


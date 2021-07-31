from flask import Flask,render_template,url_for,request
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    text_ = file_name.split(". ")
    sentences = []
    
    for sentence in text_:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2 , stopwords=None):
    if stopwords is None:
        stopwords = []
        
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0]* len(all_words)
    vector2 = [0]*len(all_words)
    
    #building vector first sentence
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] +=1
        
     # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    return 1- cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2]= sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
        return similarity_matrix
    
    

def generate_summary(file_name, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []
        
    sentences = read_article(file_name) #reading senetence and splitting it
    sentence_sim_matrix = build_similarity_matrix(sentences, stop_words) #genrate similarity matrix
        
    sentence_sim_graph = nx.from_numpy_array(sentence_sim_matrix)
    scores = nx.pagerank(sentence_sim_graph) #rank sentence in similarity matrix
        
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
        
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
            
        return ". ".join(summarize_text)
        

app = Flask(__name__)

@app.route('/')
def homepage():
	return render_template('index.html')

@app.route('/predict', methods =['POST'])
def original_text_form():
		text = request.form['text']
		summary = generate_summary(text,2)
		return render_template('result.html', summary=summary)


if __name__ == '__main__':
	app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template
import keras.models
import nltk #Natural Language tool kit
from nltk.corpus import stopwords

import re
import sys 
import os
sys.path.append(os.path.abspath("./model"))

from load import * 
app = Flask(__name__)

global model
model = init()
	

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict_api',methods=['POST' , 'GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    voc_size=5000

    messages = request.json
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', messages)
    review = review.lower()
    review = review.split()
      
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


    onehot_repr=[one_hot(words,voc_size)for words in corpus] 

    """## Embedding Representation"""

    sent_length=50
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    prediction = model.predict([np.array(embedded_docs)])

    output = prediction[0]
    return jsonify({"name" : output})
	

if __name__ == "__main__":
    app.run(debug=True)
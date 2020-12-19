# Importing the libraries

import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import nltk #Natural Language tool kit
import re #Regular Expressions
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv('train.csv')

"""Drop Nan Values"""

df=df.dropna()

"""Get the Independent Features"""

X=df.drop('label',axis=1)

""" Get the Dependent features"""

y=df['label']

"""#Data Pre-processing"""

"""Vocabulary size"""

voc_size=5000

"""###Onehot Representation"""

messages=X.copy()
messages.reset_index(inplace=True)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


onehot_repr=[one_hot(words,voc_size)for words in corpus] 

"""## Embedding Representation"""

sent_length=50
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

from tensorflow.keras.layers import Dropout

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


"""# Model"""

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

"""### Model Training"""

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


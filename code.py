import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as api
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import ssl
import io
from math import log10
from gensim.models import KeyedVectors


# Load FASTTEXT
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        data[word] = coefs
    return data
fasttext = load_vectors("cc.zh.300.vec")


## Load Skipgram
# skip_gram = KeyedVectors.load_word2vec_format("raw_data/Embeddings/tencent_chinese.txt", binary=False)


### Choose embedding
word_embedding = fasttext


## Upload subtlex
df_subtlex = pd.read_excel("SUBTLEX-CH-WF.xlsx")
subtlex_dict = {}
for i in range(len(df_subtlex.axes[0])):
    subtlex_dict[df_subtlex.iat[i,0]] = df_subtlex.iat[i,1]


### Function inputs list of words and outputs list of frequencies in SUBTLEX for each word
def get_subtlex(input_words):
    output_freqs = []
    for word in input_words:
        if word in subtlex_dict.keys():
            output_freqs.append(subtlex_dict[word])
        else:
            output_freqs.append(0)
    return output_freqs



### Upload training data
df = pd.read_excel("training_set.xlsx", header=None)
words = df.iloc[: , 0].tolist()
subtlex = get_subtlex(words)
prob = df.iloc[: , 1].tolist()




### Prepare training data
X = []
Y = []
for i in range(len(words)):
    try:
        x = list(word_embedding[words[i]])
        x.append(subtlex[i])
        X.append(np.array(x))
        Y.append(log10(prob[i]))
    except:
        pass



## Train model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X))
def build_and_compile_model(norm):
  model = keras.Sequential([
        norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss = "mean_absolute_error", metrics = "accuracy",
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
nn_model = build_and_compile_model(normalizer)
history = nn_model.fit(np.array(X), np.array(Y), verbose=0, epochs=100)



### Obtain results for Norvig
df = pd.read_excel("test_set.xlsx")
pred_words = df.iloc[:, 0].tolist()
pred_subtlex = get_subtlex(pred_words)
predict_words = []
X_predict = []
for i in range(len(pred_words)):
    try:
        x = list(word_embedding[pred_words[i]])
        x.append(pred_subtlex[i])
        X_predict.append(np.array(x))
        predict_words.append(pred_words[i])
    except:
        pass



#### Predict
predictions = nn_model.predict(np.array(X_predict))
predictions = predictions.tolist()


### Save in list of dicts
list_of_dicts = []
for i in range(len(X_predict)):
    new_dict = {}
    new_dict["word"] = predict_words[i]
    new_dict["score"] = predictions[i][0]
    ## Add only if it doesn't exist in training set
    if new_dict["word"] in words:
        pass
    else:
        list_of_dicts.append(new_dict)



## Add training set
for i in range(len(words)):
    new_dict = {}
    new_dict["word"] = words[i]
    new_dict["score"] = log10(prob[i])
    list_of_dicts.append(new_dict)



list_of_dicts = sorted(list_of_dicts, key=lambda i: i['score'], reverse=True)
df = pd.DataFrame(list_of_dicts)
with pd.ExcelWriter("ranked_lists_chinese.xlsx", engine="openpyxl") as writer: 
    df.to_excel(writer, sheet_name= "Reg new resource", index = None , header = None)
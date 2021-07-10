from flask import Flask , jsonify, render_template
from textblob import TextBlob
from polyglot.text import Text
from polyglot.text import cached_property, Text
import torch
from torch import nn, optim
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, TFBertModel
import fasttext
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pandas as pd

app = Flask(__name__)
model_name = 'bert-base-cased'

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/sentimentTextBlob/<message>')
def sentimentTextBlob(message):
	text = TextBlob(message)
	response = text.polarity
	return jsonify(response)


def maximum(a, b, c):
  
    if (a >= b) and (a >= c):
        largest = a
  
    elif (b >= a) and (b >= c):
        largest = b
    else:
        largest = c
          
    return largest


def sentiment(positive, neural, negative):
    if (positive > negative):
        return "Positive"
    elif (negative > positive):
        return "Negative"
    else:
        print("Neural")
        

@app.route('/sentimentPolyglot/<message>')
def sentimentPolyglot(message):
	neural = 0
	positive = 0
	negative = 0
	text = Text(message)
	# text2 = Text(message)
	for word in text.words:
		if (word.polarity == 0): 
			neural += 1
		elif (word.polarity == 1): 
			positive += 1
		elif (word.polarity == -1): 
			negative += 1

	print('HERE - ', positive, neural, negative)
	d = {'Positive': positive, 'Neural': neural, 'Negative': negative}
	response2 =  max(d, key=d.get)
	return jsonify(response2)


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)  #load bert model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids= input_ids, attention_mask= attention_mask, return_dict=False)
        output = self.drop(pooled_output)

        return self.out(output)

@app.route('/sentimentBert/<message>')
def sentimentBert(message):
	class_names = ['Negative', 'Positive']
	model = SentimentClassifier(len(class_names))
	model = model.to("cpu")
	model.load_state_dict(torch.load('bert_model.bin', map_location='cpu'))
	tokenizer = BertTokenizer.from_pretrained(model_name)
	device = torch.device("cpu")	
	max_len = 160
	encoded_review = tokenizer.encode_plus(
		message,
		max_length= max_len,
		truncation= True,
		add_special_tokens= True,
		return_token_type_id= False,
		pad_to_max_length=True,
		return_attention_mask= True,
		return_tensors= 'pt')

	input_ids = encoded_review['input_ids'].to(device)
	attention_mask = encoded_review['attention_mask'].to(device)

	output = model(input_ids, attention_mask)
	_, prediction = torch.max(output, dim=1)

	response = class_names[prediction]
	return jsonify(response)

def to_sentiment(sentiment):
  if sentiment == 0:
    return "Negative"
  elif sentiment == 1:
    return "Positive"


@app.route('/sentimentLSTM/<message>')
def sentimentLSTM(message):
	data = pd.read_csv('Dataset_10000.csv')
	data.dropna(subset=['sentence'], inplace=True)
	data['sentiment'] = data.sentiment.apply(to_sentiment)
	
	model = load_model('lstm_model/')
	max_features = 2000
	tokenizer = Tokenizer(num_words=max_features, split=' ')
	tokenizer.fit_on_texts(data['sentence'].values)
	text = [message]
	text = tokenizer.texts_to_sequences(text)
	text = pad_sequences(text, maxlen=29, dtype='int32', value=0)
	sentiment = model.predict(text,batch_size=1,verbose = 2)[0]
	if(np.argmax(sentiment) == 0):
		response = "Negative"
	elif (np.argmax(sentiment) == 1):
		response = "Positive"
	return jsonify(response)





@app.route('/sentimentLSTMarray/<message>')
def sentimentLSTMarray(message):
	response = []
	positive = []
	negative = []
	lists = message.split(',')
	print('message - ', lists)
	data = pd.read_csv('Dataset_10000.csv')
	data.dropna(subset=['sentence'], inplace=True)
	data['sentiment'] = data.sentiment.apply(to_sentiment)
	
	model = load_model('lstm_model/')
	max_features = 2000
	tokenizer = Tokenizer(num_words=max_features, split=' ')
	tokenizer.fit_on_texts(data['sentence'].values)
	array = lists
	print("array len - ", len(array))
	for item in range(len(array)):
		text = [array[item]]
		text = tokenizer.texts_to_sequences(text)
		text = pad_sequences(text, maxlen=29, dtype='int32', value=0)
		sentiment = model.predict(text,batch_size=1,verbose = 2)[0]
		if(np.argmax(sentiment) == 0):
			response = "Negative"
			negative.append(array[item])
		elif (np.argmax(sentiment) == 1):
			response = "Positive"
			positive.append(array[item])

	response = return_values(positive, negative)
	print('response - ', response)
	return jsonify(response)

def return_values(positive, negative):
	
	d = dict()
	d['positive'] = positive
	d['negative'] = negative
	return d


if __name__ == "__main__":
	app.run(debug=True)



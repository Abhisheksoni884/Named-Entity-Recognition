import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import spacy
from spacy import displacy
import numpy as np
import pickle


model = tf.keras.models.load_model("ner_lstm_model.h5")

with open('model_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

tokenize = model_data['tokenize']
max_len = model_data['max_len']
idx2tag = model_data['idx2tag']
tag_dict = model_data['tag_dict']
tag2idx = model_data['tag2idx']



def preprocess_sentence(sentences, tokenizer, max_len):
	# Tokenize and pad sentences as done with the training data

	seq = tokenizer.texts_to_sequences(sentences)
	padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
	seq_lengths = [len(s) for s in seq]  # Keep track of the original lengths before padding
	return padded_seq, seq_lengths

def decode_predictions(pred, seq_lengths, idx2tag):
    
	output = []
	for idx, pred_i in enumerate(pred):
		output_i = []
		length = seq_lengths[idx]  # Length of the actual sentence
		for j in range(length):  # Only iterate over the actual sentence length
			p = pred_i[j]
			p_i = np.argmax(p)
			output_i.append(idx2tag[p_i])
		output.append(output_i)
	return output


def update_model(model, new_sentences, new_labels, epochs=2):
	"""
	Updates the existing NER model with new data.

	Attributes: 

		model: The existing trained NER model.
		new_sentences: A list of new sentences for training.
		new_labels: A list of corresponding NER labels for the new sentences.
		epochs: The number of epochs to train the model with new data.

	"""

	# Preprocess the new data
	new_X, new_lengths = preprocess_sentence(new_sentences, tokenize, max_len)
	new_y_encoded = [[tag2idx[tag] for tag in doc] for doc in new_labels]  # Encode new labels
	new_y_pad = padding_sequence(new_y_encoded, max_len)
	new_Y = to_categorical_labels(new_y_pad, len(tag2idx))

	# Fine-tune the model with the new data
	model.fit(new_X, new_Y, epochs=epochs, batch_size=32) 

	return model


def padding_sequence(data,max_length):
     
	"""
	This function is used for padding the sentences

	Attributes :
		data = Encoded data that have to be padded
		max_length = Maximum length of the sentence

	return : padded data by convert into numpy array
	"""

	pad_data = pad_sequences(data, maxlen = max_length, padding ='post')
	pad_data = np.array(pad_data)
	print("The shape of the padded data is : ", pad_data.shape)
		
	return pad_data

def to_categorical_labels(data,classes):
	"""
	This function converts the labels to categorical
	"""
	categorical_data = [to_categorical(i, num_classes = classes) for i in data]
	categorical_data = np.array(categorical_data)
	print("The shape of the categorical data is : ", categorical_data.shape)
	return categorical_data



if __name__ == "__main__":

	#Load the saved model
	
	max_len = 113  
	nlp = spacy.load("en_core_web_sm")
	st.markdown("""
	<style>
		.stApp {
		background: url("https://images.hdqwalls.com/wallpapers/yellow-white-paper-4k-6o.jpg");
		background-size: cover;
		color: white;
		}
		.stButton {
		color : black;
		}
		.stSidebar{
        background-color : transparent;
        color : white;
        } 
	</style>""", unsafe_allow_html=True)

	st.title("Named Entity Recognition Model by using LSTM")

	text_input = st.text_area("",placeholder= "Enter your sentence here")
	st.sidebar.write("About")
	st.sidebar.write("""
		This NER labels are associated wuth these number according to their tag, respectively
				   
		0: Outside
				  
		1: Beginning-PER
				  
		2: Inside-PER
				  
		3: Beginning-ORG
				  
		4: Inside-ORG
				  
		5: Beginning-LOC
				  
		6: Inside-LOC
				  
		7: Beginning-MISC
				  
		8: Inside-MISC""")

	if st.button("Predict"):
		if text_input:
			# Preprocess the input
			new_X, new_lengths = preprocess_sentence([text_input], tokenize, max_len)

			# Predict the words
			predictions = model.predict(new_X)

			# Decode prediction
			decoded_predictions = decode_predictions(predictions, new_lengths, idx2tag)

			# Display results
			doc = nlp(text_input)
			entities = []
			start_char = 0
			pred = decoded_predictions[0]

			for j, token in enumerate(doc):
				label = tag_dict[pred[j]]
				if label != 'Outside':
					entities.append({
						"start": start_char,
						"end": start_char + len(token.text),
						"label": label
					})
				start_char += len(token.text) + 1

			html = displacy.render([{"text": text_input, "ents": entities, "title": "NER"}], style="ent", manual=True)
			st.markdown(html, unsafe_allow_html=True)
		

			#New data for continuous Learning section
			st.subheader("For Continuous Learning : ")
			new_sentence = st.text_input("",placeholder="Enter your Sentence for continous Learning:")
			new_label = st.text_input("",placeholder="Enter Labels as (e.g., 0 0 0 3 4):")

			if st.button("Update Model"):
				if new_sentence and new_label:
					try:
						new_labels = [int(x) for x in new_label.split()]
						#Add data to the model
						model = update_model(model, [new_sentence],[new_labels])
						st.success("Model updated!")
					except:
						st.error("Invalid label format. Please use space-separated integers.")
				else:
					st.write("please enter the sentences or labels first")


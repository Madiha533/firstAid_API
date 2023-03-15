import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import nltk
from flask import Flask,request, render_template
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
for x, doc in enumerate(docs_x):
    bag = [1 if w in doc else 0 for w in words]
    training.append(bag)

# Initialize the label encoder
le = LabelEncoder()

# Encode the class labels as integers
docs_y = le.fit_transform(docs_y)

# Convert the encoded labels to one-hot vectors
output = tflearn.data_utils.to_categorical(docs_y, nb_classes=len(labels))

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Define a Flask app
app = Flask(__name__, template_folder='templates')

# Define a route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get user input
    user_input = request.form['user_input']

    # Preprocess user input
    bag = bag_of_words(user_input, words)

    # Generate a response from the chatbot
    results = model.predict([bag])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.5:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        bot_response = random.choice(responses)

    else:
        bot_response = "I didn't get that, try again"

    # Render the response
    # return render_template('chatbot.html', user_input=user_input, bot_response=bot_response)
    return bot_response

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

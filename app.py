import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'Yes'
    else:
        output = 'No'

    return render_template('index.html', prediction_text='Your Loan Status is -- {}'.format(output))


if __name__ == '__main__':
    app.run(debug=False)

import flask
from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd
import joblib

app = flask.Flask(__name__)

model = joblib.load('models/gs_pipeline_hb_lr_tfidf_all.jlib')


def make_predictions(item):
    review = {'review_length': [len(item)], 'review_title_body': [item]}
    review = pd.DataFrame(review, index=None)
    predicted_score = int(model.predict(review.review_title_body))
    score_breakdown = model.predict_proba(review.review_title_body)[0]
    score_breakdown = [np.round(i*100, 2) for i in score_breakdown]
    return (predicted_score, score_breakdown)


# This method will forward you from the landing page to /home
@app.route('/')
def index():
    return flask.redirect(flask.url_for('home'))


# This method takes input via an HTML page
@app.route('/home', methods=['POST', 'GET'])
def home():
    return flask.render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    """Gets prediction using the HTML form"""

    if flask.request.method == 'POST':
        inputs = flask.request.form
        review = inputs['review_text']

        item = str(review)
        results = make_predictions(item)
        return flask.render_template('results.html', results=results)


if __name__ == '__main__':

    app.run(debug=True)

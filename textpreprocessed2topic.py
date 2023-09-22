from flask import Flask, request
from traceback import format_exc
import numpy as np
from bertopic import BERTopic

app = Flask(__name__)

def topic_distr(
    text: str,
    theta: float
) -> list:
    """Compute the topic distribution for text.
    text : str
        The input text on the immigration subject.

    Returns
    -------
    list
    """
    topic_model = BERTopic.load('brema76/topic_immigration_it')
    topics, probs = topic_model.transform(text)
    probs = probs/probs.sum()
    labels = topic_model.custom_labels_
    if theta>0:
        pos = np.where(probs > theta)[1]
        probs = probs[probs > theta]
        labels = [labels[i] for i in pos]
    probs = probs.tolist()
        
    return labels, probs
    

@app.route('/topic_prediction', methods=['GET'])
def get_topic():
    try:
        input_data = request.get_json()
        if "text" not in input_data:
            return {"error": "A dictionary of preprocessed texts is required in the JSON file"}, 400
        if input_data['text']['topic'] is None:
                return {"error": "The input text results to be empty and cannot be processed."}, 400
        text = input_data['text']['topic']
        theta = input_data["theta"]
        topic_prediction = topic_distr(text,theta)
        return {"topic_prediction": topic_prediction}, 200
    except Exception:
        return {"error": format_exc()}, 400

if __name__ == '__main__':
    app.run(debug=True)

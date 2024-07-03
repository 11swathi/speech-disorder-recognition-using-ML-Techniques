import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from datetime import date
import math
import librosa
import json
from playsound import playsound

app = Flask(__name__)
model = tf.keras.models.load_model('weights.h5')

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render_template('index.html', request="POST")
    else:
        return render_template("index.html")


@app.route('/sample',methods=['POST'])
def sample():
    
   
  return  playsound('tag_00023_00002135809.wav')





@app.route('/predict',methods=['POST'])
def predict():
    
    data = {
        "mfcc": []
    }
    SAMPLE_RATE = 16000
    TRACK_DURATION = 1
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    num_mfcc=13
    n_fft=2048
    hop_length=512
    num_segments=1
    json_path = "data_2.json"
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    signal, sample_rate = librosa.load( "audio.wav", sr=SAMPLE_RATE)
    if len(signal) >= SAMPLE_RATE:
        signals = signal
    else:
        signal = np.pad(
            signal,
            pad_width=(SAMPLE_RATE - len(signal), 0),
            mode="constant",
            constant_values=(0, 0),
        )
    for d in range(num_segments):

        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            print("segment:{}".format( d+1))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    with open(json_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    X = X[..., np.newaxis]
    B = X[0]
    Y = B[np.newaxis, ...]
    prediction = model.predict(Y)
    predicted_index = np.argmax(prediction, axis=1)    
    if predicted_index == 0:
        out =" You have Dysarthria "
    if predicted_index == 1:
        out =" Your speech is Normal"
    if predicted_index == 2:
        out =" You have stuttering"
   

   

    return render_template('index.html', prediction_text=' {} '.format(out))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

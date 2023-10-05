from io import BytesIO
import base64
import scipy as scipy
from flask import Flask, request, render_template, send_file
from flask import jsonify

from setuptools._entry_points import render
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import h5py
import tensorflow as tf
import os
import tensorflow_addons as tfa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

app = Flask(__name__)

model = load_model("weights_28_0.06.hdf5", compile=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_page")
def func():
    return render_template("Upload_Img.html")


@app.route("/upload_page_2")
def func2():
    return render_template("Upload_2_img.html")


def attentionmap(img, csv):
    new_in = np.zeros((csv.shape[0], csv.shape[1], 4))
    new_in[:, :, 0] = img[:, :, 0] / 255.0
    new_in[:, :, 1] = img[:, :, 1] / 255.0
    new_in[:, :, 2] = img[:, :, 2] / 255.0
    new_in[:, :, 3] = csv / np.amax(csv)
    attention_layer = tf.keras.models.Model(inputs=model.inputs,
                                            outputs=model.get_layer('multi_head_attention_7').output)
    multi_head_output, attention_maps = attention_layer.predict(img)
    final_attention_map1 = tf.reduce_min(attention_maps[0][0], axis=1)
    final_attention_map_1 = tf.reshape(final_attention_map1, (30, 40))
    importance_matrix_1 = scipy.ndimage.zoom(final_attention_map_1, 16, order=0)
    return importance_matrix_1, csv

    # new_in = np.zeros((csv.shape[0], csv.shape[1], 4))
    # new_in[:, :, 0] = img[:, :, 0] / 255.0
    # new_in[:, :, 1] = img[:, :, 1] / 255.0
    # new_in[:, :, 2] = img[:, :, 2] / 255.0
    # new_in[:, :, 3] = csv / np.amax(csv)
    # # Add a fourth channel of zeros
    # new_in = np.concatenate([new_in, np.zeros((new_in.shape[0], new_in.shape[1], 1))], axis=-1)
    # result = float(model(np.array([new_in]))[0][0].numpy())


@app.route('/Upload_Img', methods=['POST'])
def predict():
    img = request.files['img']
    csv = np.loadtxt(request.files['csv_file'], delimiter=",")
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
    new_in = np.zeros((csv.shape[0], csv.shape[1], 4))
    new_in[:, :, 0] = img[:, :, 0] / 255.0
    new_in[:, :, 1] = img[:, :, 1] / 255.0
    new_in[:, :, 2] = img[:, :, 2] / 255.0
    new_in[:, :, 3] = csv / np.amax(csv)
    result=float(model(np.array([new_in]))[0][0].numpy())
    rounded_result = round(result, 2)
    # value = "week"
    # rounded_result = {"prediction": rounded_result , "text": value}
    return jsonify({"prediction": str(rounded_result)+ " week"})


@app.route('/upload_2_img', methods=['POST'])
@app.route('/upload_2_img', methods=['POST'])
def plot():
    img = request.files['img']
    csv = np.loadtxt(request.files['csv_file'], delimiter=",")
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
    new_in = np.zeros((csv.shape[0], csv.shape[1], 4))
    new_in[:, :, 0] = img[:, :, 0] / 255.0
    new_in[:, :, 1] = img[:, :, 1] / 255.0
    new_in[:, :, 2] = img[:, :, 2] / 255.0
    new_in[:, :, 3] = csv / np.amax(csv)
    i1, csv = attentionmap(new_in, csv)
    # generate the plot
    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(new_in[:,:,:3])

    ax2 = fig.add_subplot(2,2,3)
    ax2.imshow(new_in[:,:,:3])
    ax2.imshow(i1, cmap='jet', alpha=0.3)
    ax2.axis("off")
    plt.colorbar()

    # convert the plot to a PNG image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # send the image as a response to the client
    return jsonify({"img": base64.b64encode(buf.read()).decode("ascii")})
# def plot():
#     img = request.files['img']
#     csv = np.loadtxt(request.files['csv_file'], delimiter=",")
#     img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
#     new_in = np.zeros((csv.shape[0], csv.shape[1], 4))
#     i1, csv = attentionmap(img, csv)
#     # generate the plot
#     fig = plt.figure(figsize=(10,10))
#
#     ax1 = fig.add_subplot(2,2,1)
#     ax1.imshow(new_in[:,:,:])
#
#     ax2 = fig.add_subplot(2,2,3)
#     ax2.imshow(new_in[:,:,:3])
#     ax2.imshow(i1, cmap='jet', alpha=0.3)
#     ax2.axis("off")
#     plt.colorbar()
#
#     # convert the plot to a PNG image
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     # send the image as a response to the client
#     return jsonify({"img": base64.b64encode(buf.read()).decode("ascii")})

if __name__ == "__main__":
    app.run(debug=True, port=5500)

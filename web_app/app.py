from flask import Flask, render_template, request
from torch_utils import transform_image, prediction
import os

app = Flask(__name__)
UPLOAD_FOLDER = './static/images/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods=['GET'])
def home() : 
    return render_template('index.html')

@app.route('/', methods=['POST']) 
def predict() :

    if 'imagefile' not in request.files:
        return render_template('index.html', predict=None)

    img_file = request.files['imagefile']

    if img_file.filename == '':
        return render_template('index.html', predict=None)

    # simpan file dulu
    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    # buka ulang untuk prediksi
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    input_tensor = transform_image(image_bytes)
    yhat = prediction(input_tensor)

    # kirim path relatif ke template
    return render_template('index.html',
                           predict=str(yhat), 
                           image_url=f"/{img_path}")


if __name__ == '__main__' : 
    app.run(port=3000, debug=True)

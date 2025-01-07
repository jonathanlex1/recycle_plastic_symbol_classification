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

    if 'imagefile' in request.files :
    #input image 
        img_file = request.files['imagefile'] #imagefile sesuaikan dengan name pada input html

        if img_file.filename != '' :
           
            #predict image
            image_bytes = img_file.read()
            input_tensor = transform_image(image_bytes)
            yhat = prediction(input_tensor)
            
            #saving the image
            img_path = os.path.join(UPLOAD_FOLDER ,img_file.filename)
            img_file.save(img_path)

    return render_template('index.html', predict=yhat)



if __name__ == '__main__' : 
    app.run(port=3000, debug=True)
    


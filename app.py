from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

model = load_model('model/modelTomat.h5')
labels = ['matang', 'mentah', 'setengah matang']
images = 'static/uploads'  
os.makedirs(images, exist_ok=True) 

def classify(imagepath):
    image = load_img(imagepath, target_size=(128,128))
    image_array = img_to_array(image)/255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = str(str(int(prediction[0][index]*100))+ '%')

    return class_name, confidence_score

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        imagefile= request.files['imagefile']
        imagepath = os.path.join(images, imagefile.filename)
        imagefile.save(imagepath)

        class_name, confidence_score = classify(imagepath)

        result = {
            'kelas': class_name,
            'confidence': confidence_score,
            'image': imagepath
        }

    return render_template('index.html', result=result)


    

if __name__ == '__main__':
    app.run(debug=True)
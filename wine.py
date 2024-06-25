from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('wine.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
log = objects['model']
scaler= objects['scaler']

@app.route('/')
def homepage():
    return render_template('wine.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    a = request.form.get('volatile acidity')
    b = request.form.get('citric acid')
    c = request.form.get('chlorides')
    d = request.form.get('total sulfur dioxide')
    e = request.form.get('density')
    f = request.form.get('sulphates')
    g = request.form.get('alcohol')
    
    x = np.array([[a,b,c,d,e,f,g,]])
    
    x = scaler.transform(x)
    pred = log.predict(x)
    msg = 'The quality of wine is good' if pred == 1 else 'The quality of wine is bad'
    
    return render_template('wine.html',text=msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
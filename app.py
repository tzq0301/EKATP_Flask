import os

import torch
from flask import Flask, render_template, request, send_file

import gene.gene
import predict.predict as pred
from model.model import embed_koopmanAE

PORT: int = 5000

APP_ROOT: str = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER: str = os.path.join(APP_ROOT, 'static', 'temp')
DATA_FOLDER: str = os.path.join(APP_ROOT, 'static', 'datas')
RESULT_FOLDER: str = os.path.join(APP_ROOT, 'static', 'results')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

MODEL_PATH: str = './static/models/model.pkl'
# model = torch.load(MODEL_PATH)  # 使用 with torch.no_grad(): ...
model = embed_koopmanAE(84, 1, 8, 14, 12, 1, 0.99)
model.load_state_dict(torch.load(MODEL_PATH))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', message='Hello EKATP!!!')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        try:
            f = request.files['file']
            full_filename: str = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(full_filename)
            step = request.form.get("step")
            filename: str = pred.predict_(model, full_filename, app.config['RESULT_FOLDER'],
                                          step=int(step) if step != '' else 6)
            return send_file(filename)
        except Exception:
            return render_template('501.html')


@app.route('/paper')
def paper():
    return render_template('paper.html')


@app.route('/example', methods=['GET', 'POST'])
def example():
    if request.method == 'POST':
        f = request.files['file']
        full_filename: str = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(full_filename)
        img_name: str = gene.gene.example(model, full_filename)
        return render_template('example.html', img_name=img_name)
    elif request.method == 'GET':
        img_name: str = gene.gene.example(model, os.path.join(app.config['DATA_FOLDER'], 'gene.csv'))
        return render_template('example.html', img_name=img_name)


@app.route('/img', methods=['GET'])
def img():
    IMG_NAME: str = 'results/demo.jpg'
    return render_template('img.html', img_name=IMG_NAME)


@app.route('/data-gene-csv', methods=['GET'])
def data_gene_csv():
    filename: str = os.path.join(app.config['DATA_FOLDER'], 'gene.csv')
    try:
        return send_file(filename)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(port=PORT)

import flask
import os
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.io import read_image
import csv

from model_training import image_model

app = flask.Flask(__name__)

global label_dict
global model
global device


def read_dict_csv(filepath):
    csv_dict = {}
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                key, value = row
                csv_dict[key] = value
    return csv_dict


def init_model(weights_path, dict_path):
    global label_dict
    global model
    global device

    if not os.path.exists('model.pth'):
        print('Model not found. Creating training and validation splits...')
        if not os.path.exists('model_training/data/'):
            print('Error: no data has been found. Please place a folder named "data" in this directory and try again. ')
            exit(500)
        os.system('python dataset_builder.py')
        print('Training and validation splits created. Training model...')
        label_dict = read_dict_csv(dict_path)
        image_model.train_model(num_classes=len(label_dict))
        print('Model training complete.')

    # create class reference dictionary
    label_dict = read_dict_csv(dict_path)

    # assign computation device and initialize model frame
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using: {device}")
    model = image_model.CustomEfficientNet(len(label_dict))
    model = model.to(device)

    # load model weights and set to evaluation mode

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()



@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Check for JSON or form data
    if flask.request.is_json:
        params = flask.request.json
    else:
        params = flask.request.args

    if 'file_name' not in params:
        return flask.jsonify({'error': 'file_name not provided'}), 400

    try:
        image_path = params['file_name']
        image_tensor = read_image(image_path)

        transform = models.EfficientNet_V2_S_Weights.DEFAULT.transforms()

        image_tensor = transform(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        output = model(image_tensor)
        output = F.softmax(output, dim=1)
        mold = output.cpu().detach().numpy()[0]

        data = {label_dict[str(i)]: str(mold[i - 1]) for i in range(1, len(label_dict) + 1)}
        json_label = f'x1: 0, y1: 0, x2: {image_tensor.size(dim=2)}, y2: {image_tensor.size(dim=3)}'

        response = {str(json_label): data}
        return flask.jsonify(response)
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500



import argparse

# Required for transform to function properly
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
os.chdir(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Process some input paths')
parser.add_argument('--model_path', type=str, default='model.pth', nargs='?',
                    help='path to the model based on working directory')
parser.add_argument('--label_path', type=str, default='label_dict.csv', nargs='?',
                    help='path to the labels based on working directory')
parser.add_argument('--port', type=int, default=5000, nargs='?',
                    help='port you want to use')
args = parser.parse_args()
# model_path = args.model_path
# dictionary_path = args.label_path
port = args.port
model_path = args.model_path
dictionary_path = args.label_path

init_model(model_path, dictionary_path)

# start the flask app, allow remote connections
app.run(host='0.0.0.0', port=port)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


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


def train_model(data_filepath, dict_path):
    global label_dict

    os.system('python model_training/dataset_builder.py --data ' + data_filepath)
    label_dict = read_dict_csv(dict_path)
    l1_reg = input("Enter a ")
    image_model.train_model(len(label_dict))


def init_model(weights_path, dict_path):
    global label_dict
    global model
    global device

    if not os.path.exists('model.pth'):
        print('Error: "model.pth" not found. Please place training data in this directory (or designate the data location with --data_path) and run '
              '"python main.py --train" in the console.')
        exit(500)

    # create class reference dictionary
    label_dict = read_dict_csv(dict_path)

    # assign computation device and initialize model frame
    device = image_model.find_device()
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
parser.add_argument('--data_path', '-d', type=str, default='data/', nargs='?',
                    help='path to the training data')
parser.add_argument('--port', type=int, default=5000, nargs='?',
                    help='port you want to use')
parser.add_argument('--train', '-t', action='store_true', help="Set this flag if you want the model to train itself using present data")
args = parser.parse_args()
# model_path = args.model_path
# dictionary_path = args.label_path
port = args.port
model_path = args.model_path
dictionary_path = args.label_path
data_path = args.data_path

if args.train:
    print("Application started in training mode. Once this is complete, any model.pth will be rewritten if it exists.")
    answer = ''
    valid_answers = {'y', 'n'}
    while answer not in valid_answers:
        answer = input('Are you sure you want to continue? (y/n)')
    if answer == 'n':
        exit(500)
    if not os.path.exists(data_path):
        print('Error: no data has been found. Please place a folder named "data" in this directory (or designate the data location with --data_path) and try again. ')
        exit(500)
    train_model(data_path, dictionary_path)


init_model(model_path, dictionary_path)

# start the flask app, allow remote connections
app.run(host='0.0.0.0', port=port)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


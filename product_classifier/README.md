# Op-Smart AI Suite

This project is the complete suite of AI tools used by Op-Smart, Inc. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Product Classifier](#product-classifier)
- [Image Splitter](#image-splitter)
- [QA Evaluator](#qa-evaluator)
- [Usage](#usage)
- [License](#license)

## Introduction

This project has two pieces: the server, and the model trainer.

On startup, the server will check for the stored model weights. If they are present, the model will be initialized and the server will start running. If
they are not present, the server will check for the presence of image data, which it will start training on if present. 

If the model weights are detected, the server will initialize the model with those weights and start waiting for prediction requests. 
Any prediction request should follow this format:

```bash
   curl --data-raw "{\"file_name\":\"/your/file/path\"}" -H "Content-Type: application/json" -L -X POST http://localhost:5000/predict >/your/file/path.txt
```

## Installation

To get started, <b>you need to have Python 3.10+ installed.</b> The installer for Python 3.11, which fulfills this requirement, can be downloaded by [clicking this link.](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) You can set up the project using the following steps:


1. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
2. **Run the classification model:**

   ```bash
   # In the product_classifier folder:
   python main.py
   ```
   Running the model once in a standalone command prompt or PowerShell instance before production usage is very important. 
   If the model has not been trained yet or there is not a file named "model.pth" present in the package directory, the package will attempt to train a new model.
   **If there is no data in the root folder when this happens, THE CLASSIFICATION SERVER WILL NOT START**. There must either be a "model.pth" file
   or training data present for the server to function correctly. 

## Product Classifier

1. **Training:**

   On the first startup of the classification server, once data is present, the model will begin training. It will ask for a **regularization coefficient** in the console, and wait for a user input.
   The regularization coefficient (hereby referred to as λ) is a hyperparameter: a parameter that will affect the training process of the model. 
   It is meant to help adjust the training process so that it can run on multiple dataset sizes.

   **A good λ to start at is 1e-6**. Enter that, then let the model begin training. If the first loss value reported is above 15, restart the program and enter a slightly lower λ . If the first loss value reported is below 10, restart the program and enter a slightly higher λ.
   For the model to train properly, you need the first loss value to be between 10 and 15. 
   
   *For example*: on a dataset of 2600 images, the model reached 100% accuracy with λ = 2e-5.

## Image Splitter

## QA Evaluator

## Usage

To start the server, run:

```bash
python server.py
```

## License

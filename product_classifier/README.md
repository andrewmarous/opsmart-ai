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

This project is a collection of every AI tool used in the Op-Smart InLine product assessment and analytics software.

There are three components: the product classifier, the image splitter, and the quality assurance evaluator. Please see their respective sections for more information about each.

## Installation

To train the model, <b>you need to have Python 3.10+ installed.</b> The Windows installer for Python 3.11, which fulfills this requirement, can be downloaded by [clicking this link.](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) You can set up the project using the following steps:


1. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
2. **Run the classification model in training mode:**

   ```bash
   # In the product_classifier folder:
   python main.py --train
   # OR (if training data not in product_classifier folder)
   python main.py --train --data "C:\path\to\your\data"
   ```
   This will train the model to work with a specific dataset. Without training the model first, the server will not work.

   During training, you will be asked for a *regularization coefficient* (see [Product Classifier](#product-classifier) for more information).
   Once that is entered, the model will begin training and stop once it reaches 100% accuracy. The server will start running once training has completed.


3. **Compile server into executable:**
   ```bash
   pyinstaller --add-data "model.pth" --add-data "label_dict.csv" --onefile main.py
   ```
   **If you are installing this program on a machine without Python, you need to do this step.** The server will not start up or run properly if an environment is not 
   set up on the target computer, so any usage of this model on a non-development machine must be through a standalone executable. Once the process has completed, you can find the executable at this subdirectory: 
   ```text
   dist\main.exe
   ```

   *Note: PyInstaller creates an executable based on the OS it is running on, so this command must be run on a computer with the same OS as the target machine. *

## Product Classifier

1. **Training:**

   1. If the server is run in training mode (see [usage](#usage)), the model will begin training. It will ask for a **regularization coefficient** in the console, and wait for a user input.
   The regularization coefficient (hereby referred to as λ) is a hyperparameter: a parameter that will affect the training process of the model. 
   It is meant to help adjust the training process so that it can run on multiple dataset sizes.

   2. **A good λ to start at is 1e-6**. Enter that, then let the model begin training. If the first loss value reported is above 15, restart the program and enter a slightly lower λ . If the first loss value reported is below 10, restart the program and enter a slightly higher λ.
   For the model to train properly, you need the first loss value to be between 10 and 15. 
   
   3. *For example*: on a dataset of 2600 images, the model reached 100% accuracy with λ = 2e-5.
2. **Prediction:**
   The model's response to an assessment request should look like this:
   ```text
   {"x1: 0, y1: 0, x2: image_width, y2: image_height":
   {"class_1":"0.037525166","class_2":"0.021139832","class_3":"0.09072289",...
   ```
   If the model returns a 400 Bad Request, there is something wrong with the curl call that requests an assessment (see [usage](#usage) for the correct call).
   If the model returns something other than either of the previous responses, there is an error within the product classifier. If it persists, please reach out to Op-Smart technical support. 
   

## Image Splitter

## QA Evaluator

## Usage

To start the server, run:

```bash
python main.py
```

To start the server in **training mode**, run:
```bash
python main.py --train

# Note: you can use the --data tag to specify the filepath to the root
# folder of your data, with the current directory being the default. Example:
python main.py --train --data "C:\Path\to\your\data"
```

All assessment requests should be formatted as so:
```bash
   curl --data-raw "{\"file_name\":\"/your/file/path\"}" -H "Content-Type: application/json" -L -X POST http://localhost:5000/predict >/your/file/path.txt
   # Note: if this call does not function properly as a Shell command, enter it in cmd instead
```

## License

This package has been designed for, and solely for, Op-Smart, Inc. and Andrew Marous. Any usage of these packages outside of
Op-Smart must be accompanied by written permission for their from the company. 

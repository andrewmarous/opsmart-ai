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

To get started, you need to have Python 3.8+ installed. You can set up the project using the following steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/andrewmarous/opsmart-ai.git
    cd product_classifier
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Product Classifier

## Image Splitter

## QA Evaluator

## Usage

To start the server, run:

```bash
python server.py
```

## License

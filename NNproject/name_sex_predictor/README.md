## Japanese Name Sex Predictor

This repository contains two main components: a Python package for training and predicting the sex (male/female) of Japanese names, and a simple web front‑end for making predictions in the browser.

### Python scripts

* **train_model.py** — Train a weight vector (`sigma`) from a dataset of names. The dataset should be provided in JSON format, where each entry has a `name` (e.g., `"Haruto Sato"`) and a `sex` (`"male"` or `"female"`). The script saves the trained weight vector to a JSON file (`model.json` by default).
* **predict.py** — Predict the sex of a given name using a previously trained `sigma`. It loads `model.json`, encodes the input name, computes the dot product with the weight vector, and outputs the prediction.

The encoding scheme reserves 12 characters each for the surname and given name (24 letters total). Each character is encoded into 5 bits with `A=00001`, `B=00010`, and so on. Shorter names are padded on the left with zeros. See the docstrings in `train_model.py` for details.

### Web application

The `webapp` directory contains a self‑contained HTML interface for making predictions in the browser:

* **index.html** — A web page with a form where the user can enter a name in capital letters and receive a prediction.
* **script.js** — Implements the encoding and prediction logic in JavaScript. It fetches the weight vector from `model.json` on load. If the file cannot be fetched, all weights default to +1 (so the page will always predict "male").
* **model.json** — A placeholder weight vector consisting of all `+1`. You can replace this file with a trained model by running `train_model.py` on your dataset and copying the resulting `model.json` into the `webapp` directory.

### Usage

1. **Training a model**

   ```bash
   python train_model.py -i data.json -o webapp/model.json
   ```

   Replace `data.json` with your own dataset (the file may also be named `data.jason`). The script will produce `model.json` with the trained weights, which can then be used both by `predict.py` and the web front‑end.

2. **Predicting from the command line**

   ```bash
   python predict.py "HARUTO SATO" -m webapp/model.json
   ```

3. **Using the web interface**

   Open `webapp/index.html` in a browser. Enter a name in capital letters, e.g., `HARUTO SATO`. The page will display the predicted sex after you click "Predict".

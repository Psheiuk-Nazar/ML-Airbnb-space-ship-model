Airbnb Ship Detection Kaggle Project

Overview
This repository contains the code and resources for the Airbnb Ship Detection Kaggle competition.
The goal of this competition is to develop an algorithm to identify and locate ships in satellite images.


Competition Link: https://www.kaggle.com/competitions/airbus-ship-detection

#Project Structure

The project is organized as follows:

requirements.txt: Lists the Python dependencies for the project.
README.md: Documentation for the project.

Getting Started
Follow these steps to get started with the project:

Clone the repository:
1. git clone https://github.com/your-username/airbnb-ship-detection.git
2. cd airbnb-ship-detection

Install the required dependencies:
3. pip install -r requirements.txt

Download the dataset from the Kaggle:
4. Made account in Kaggle
5. Generete API key
6. add API Key file in .kaggle/kaggle.json
7. kaggle competitions download -c airbus-ship-detection

Unzip documents

8. run unzip_documents

Make a model

9. Run training.ipynb

You will have a model name seg_model.h5 in the end of the run

10. Test a model run model_inference.ipynb

Results
You will have all result in validation.png files

Author
HENRIQUE MENDONÇA
Kaggle Profile: https://www.kaggle.com/hmendonca
K Scott Mader
https://www.kaggle.com/kmader
https://www.kaggle.com/code/hmendonca/u-net-model-with-submission
Author python format of ML:
Nazar Psheiuk
Kaggle Profile: https://www.kaggle.com/psheiuk
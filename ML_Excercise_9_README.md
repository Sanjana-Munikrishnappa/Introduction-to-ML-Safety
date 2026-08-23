# ML Exercise 9 — Machine Learning Safety
This notebook continues the machine learning safety experiments using the CARLA dataset and the trained traffic light, pedestrian, and vehicle classifiers from the previous exercises. The exercise focuses on evaluating how the models behave in different situations and using the results to understand their reliability and potential safety issues.

## What it does
The notebook loads the previously trained models and the required CARLA test datasets before running the models on different image samples. The predictions are collected and compared with the expected labels to examine how the classifiers perform. The notebook also analyzes the model outputs in more detail so that differences in performance and incorrect predictions can be identified.

## Requirements
The notebook is designed to run in Google Colab and uses Python libraries including torch, torchvision, pandas, matplotlib, and Pillow. The trained model files and CARLA datasets from the previous exercises are also required to reproduce the results.

## Dataset location
The notebook uses the CARLA dataset stored in /content/drive/MyDrive/carla_dataset/. The required test data and the saved model files from the earlier exercises need to be available in Google Drive so that they can be loaded during the evaluation.

## Output
The notebook performs setup and dataset checks before running the main experiments. The trained classifiers are loaded and used to generate predictions on the selected test images. The resulting predictions and evaluation information are displayed throughout the notebook to provide a detailed view of how the models behave.

**Model analysis**
The results are used to examine the behavior of the traffic light, pedestrian, and vehicle classifiers. Correct predictions show situations where the models are able to recognize the expected objects, while incorrect predictions provide examples of where the models may struggle. Examining these cases helps reveal weaknesses that may not be obvious from a single overall performance score.

**Safety analysis**
The exercise uses the model results to think about reliability from a machine learning safety perspective. A model that performs well on normal test examples can still make mistakes in particular situations, so examining its behavior across different examples is important when considering its use in real-world applications such as autonomous driving.

**What you're left with**
The notebook produces a collection of model predictions, evaluation results, and analysis that can be used to understand the reliability of the trained classifiers. The results provide a practical view of how machine learning models can be tested for potential weaknesses and safety concerns, especially when the models are being considered for applications where incorrect predictions could have real-world consequences.

# ML Exercise 8 — Model Evaluation and Safety Analysis
This notebook evaluates the three pre-trained classifiers from the previous exercises — traffic light, pedestrian, and vehicle — to understand how well they perform on the CARLA test dataset. The exercise focuses on examining the model predictions and identifying correct and incorrect classifications to better understand the reliability and safety of the trained models.

## What it does

The notebook loads the saved model weights from the previous exercises and prepares the test dataset for evaluation. Each of the three models is used to make predictions on test images, and the predictions are compared with the corresponding ground-truth labels. The results are then analyzed to understand how the models perform across the different object classes and to identify examples where the models make incorrect predictions.

## Requirements
The notebook is designed to run in Google Colab and requires Python libraries including torch, torchvision, pandas, matplotlib, and Pillow. The trained models and datasets also need to be available in Google Drive before running the notebook.

## Dataset location

The notebook uses the CARLA dataset stored at /content/drive/MyDrive/carla_dataset/. The dataset contains the test images used for evaluating the traffic light, pedestrian, and vehicle classifiers. The saved model files from the previous exercises are also loaded from the same Google Drive location.
## Output
Setup and sanity checks are performed first to make sure that the required dataset folders, images, and trained model files are available. The notebook then loads each classifier and runs it on the test images. The predictions and corresponding labels are used to evaluate the performance of the three models.

**Model evaluation**
The traffic light, pedestrian, and vehicle models are evaluated separately using their respective test images. The notebook compares the predicted class with the actual class and records the results. This makes it possible to identify which images are classified correctly and which images are misclassified by each model.

**Error analysis**
The incorrect predictions are examined to understand where the models may have difficulty. Looking at these mistakes can provide useful information about potential weaknesses in the classifiers, such as confusing similar objects or failing to recognize objects under certain visual conditions.

**What you're left with**
The notebook produces a collection of model predictions and evaluation results for the traffic light, pedestrian, and vehicle classifiers. These results provide a better understanding of how reliably the models perform on the test data and help identify potential weaknesses that could be important from a machine learning safety perspective.

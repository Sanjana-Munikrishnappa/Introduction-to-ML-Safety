# ML Exercise 4 — Traffic Light, Pedestrian & Vehicle Detection (with Evaluation & Saving)
This notebook builds on Exercise 3: it trains the same three image classifiers on the CARLA dataset (traffic light, pedestrian, vehicle detection), but adds proper evaluation metrics and saves the trained models to Google Drive.
## What it does
The dataset is mounted from Google Drive and extracted, and labels are loaded into pandas DataFrames. A custom DrivingDataset class loads each image, resizes it, and pairs it with the correct label column (has_traffic_light, has_pedestrian, or has_vehicle). The same small CNN architecture (TrafficLightClassifier: three convolution + pooling blocks followed by fully connected layers) is trained separately for each of the three targets using BCEWithLogitsLoss and the Adam optimizer. After training, each model is evaluated with precision, recall, F1-score, and a confusion matrix, then saved to Google Drive as a .pth file.
## Requirements
Google Colab, plus torch, torchvision, pandas, matplotlib, Pillow, and scikit-learn.
## Dataset location
/content/drive/MyDrive/carla_dataset/ containing train.zip, validation.zip, and test.zip.
## Output
Running the notebook produces output at several stages, mostly printed to the console or shown as inline plots and charts.

**During training (once per model):** each epoch prints its average loss, for example Epoch 1/3, Loss: 0.3618, so you can watch the loss decrease as the model learns. A loss-curve plot is shown after training finishes for each model.

**During evaluation (once per model):** three numbers are printed — precision, recall, and F1-score — measuring how well the model identifies the target class on unseen test images. A confusion matrix plot is also displayed, breaking down correct and incorrect predictions into true/false positives and negatives, which is more informative than accuracy alone, especially if the classes are imbalanced.

**During saving:** each model's weights are written to Google Drive as a .pth file (traffic_light_model.pth, pedestrian_model.pth, vehicle_model.pth), with a confirmation message printed after each save. At the end, the notebook lists every .pth file found in the save directory to verify all three were saved correctly.

**What you're left with:** three trained models with their loss curves, precision/recall/F1 scores, and confusion matrices, saved on Google Drive as .pth files so they can be reloaded later without retraining.

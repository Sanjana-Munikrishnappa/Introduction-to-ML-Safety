# ML Exercise 7 — Out-of-Distribution Detection (MSP vs. kNN)
This notebook loads the pre-trained traffic light model from Exercise 4 and tests two techniques for detecting when an input image comes from conditions the model wasn't trained on: Maximum Softmax Probability (MSP), based on the model's own confidence, and k-Nearest Neighbors (kNN), based on how unusual an image's internal features look compared to normal training-like data.
## What it does
Images from four conditions — sunny (in-distribution), fog, night, and a different town layout — are loaded and run through the traffic light model to collect confidence scores. These scores are used to compute an AUROC for MSP-based OOD detection, treating sunny as "normal" and the other three conditions combined as "abnormal." Separately, internal feature vectors are extracted from the model for the same images, and a kNN-based distance score is computed and used to calculate a second, comparable AUROC for the feature-based approach.
## Requirements
Google Colab, plus torch, torchvision, pandas, numpy, matplotlib, Pillow, and scikit-learn (for NearestNeighbors and roc_auc_score).
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing test.zip, test-fog.zip, test-night.zip, and test-town-01.zip, plus the saved traffic_light_model.pth from Exercise 4.
## Output
**Dataset checks:** file and folder listings confirm each zip archive extracted correctly, and the number of images found is printed for each of the four conditions after listing filenames.

**Sample image grids:** 5 sample images are displayed for each condition individually, labeled "Sunny," "Fog," "Night," and "Town," giving a visual sense of how different each condition looks.

**Model inspection:** the traffic light checkpoint's saved keys and the shape of every weight tensor are printed when first loaded, followed by the full model architecture. A single image is traced step by step through the network, with the tensor shape printed after each major layer, showing how the image's dimensions shrink through the CNN.

**Single-prediction output:** for one sample image, the model's raw output score is printed, followed by the same value converted into a 0–1 probability using sigmoid. A grid of 5 sunny images is then displayed with each one's predicted probability shown as its title.

**Score distributions:** confidence scores are collected for the traffic light model across all four conditions, and the count of scores gathered per condition is printed. Four histograms (one per condition) are then plotted, showing how the model's confidence scores are distributed — sunny scores typically cluster near 0 or 1, while fog/night/town scores tend to be more spread out.

**MSP AUROC:** using sunny scores as "normal" and the combined fog/night/town scores as "abnormal," a single AUROC value is printed, for example MSP AUROC = 0.8421. A value closer to 1 means the model's own confidence score is a strong signal for spotting out-of-distribution images; closer to 0.5 means it's no better than random guessing.

**kNN AUROC:** feature vectors are extracted from all four conditions, and each image's distance to its nearest neighbor among the sunny (in-distribution) features is computed. Using these distances the same way as the confidence scores above, a second AUROC value is printed, for example kNN AUROC = 0.9013. Comparing this number to the MSP AUROC shows which of the two approaches is a more reliable way to flag OOD input for this model — a higher kNN AUROC would suggest the model's internal features carry more useful signal about unfamiliar conditions than its output confidence alone.

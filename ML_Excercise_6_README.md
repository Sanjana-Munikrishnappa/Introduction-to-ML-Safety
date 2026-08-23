# ML Exercise 6 — Model Interpretability with Saliency Maps
This notebook loads the three pre-trained classifiers from Exercise 4 (traffic light, pedestrian, vehicle) and generates saliency maps to visualize what each model is "looking at" when it makes a prediction — including on correct predictions, on mistakes, and on images captured in unfamiliar conditions like fog, night, and a different town layout.
## What it does
Each model's saved weights are loaded, and an updated dataset class fixes an earlier filename-formatting bug (frame numbers need to be zero-padded, e.g. 000001.jpg, or images fail to load). For each of the three models, a handful of correctly classified test images are found and a saliency map is generated for each — computed by backpropagating the model's prediction back to the input image and taking the largest gradient magnitude per pixel across color channels. A few traffic-light images the model gets wrong are also visualized this way. Finally, the traffic light model is run on one sample each from fog, night, and alternate-town datasets to see how its attention shifts outside normal conditions.
## Requirements
Google Colab, plus torch, torchvision, pandas, matplotlib, and Pillow.
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing test.zip, test-fog.zip, test-night.zip, and test-town-01.zip, along with the saved model files traffic_light_model.pth, pedestrian_model.pth, and vehicle_model.pth from Exercise 4.
## Output
**Setup and sanity checks:** dataset sizes and directory listings are printed after extraction to confirm the test, fog, night, and town folders and their rgb-front image subfolders are structured as expected. A list of .pth files found in Drive confirms all three trained models are available before loading.

**Saliency maps for correctly classified samples:** for each of the three models, a handful of side-by-side image pairs are shown — the original photo next to a "hot" colormap heatmap highlighting the pixels that most influenced that correct prediction. Brighter areas indicate stronger influence on the model's decision.

**Saliency maps for misclassified samples:** a few more image pairs are shown for the traffic light model specifically, but using images it got wrong. Comparing these to the correct-prediction heatmaps can reveal whether the model was focusing on irrelevant parts of the scene (like the road, sky, or background) rather than the actual traffic light when it made a mistake.

**Saliency maps under unfamiliar conditions:** one more image-pair is generated and displayed for each of the fog, night, and alternate-town datasets, showing how the traffic light model's attention shifts when the visual conditions differ from what it was trained on — for example, whether it still focuses on the traffic light in low light or fog, or gets distracted by noise in the image instead.

**What you're left with:** a collection of visual (image + heatmap) pairs across normal, incorrect, and out-of-distribution cases, giving a qualitative sense of how trustworthy each model's decision-making process looks — this notebook doesn't compute a final accuracy number itself, since that quantitative comparison across conditions is the focus of a separate exercise.

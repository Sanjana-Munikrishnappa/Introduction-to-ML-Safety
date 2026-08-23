# ML Exercise 5 — Confidence Calibration & Backdoor Attack on a Pedestrian Detector
This notebook trains a ResNet18-based pedestrian detector on the CARLA dataset, then uses it to explore two ML safety topics: how well-calibrated its confidence scores are (via temperature scaling), and how vulnerable it is to a real backdoor attack (via data poisoning).
## What it does
A ResNet18 model (pretrained on ImageNet, with its final layer replaced for binary classification) is trained for 5 epochs to detect pedestrians, then evaluated on the test set. Its raw output logits are collected and passed through temperature scaling at several values (0.5, 1.0, 2.0) to see how this affects prediction confidence and accuracy. Separately, a red-square trigger is added to 10% of pedestrian-positive training images (with their label flipped to "no pedestrian"), and the model is fine-tuned on this poisoned dataset. The backdoored model is then evaluated for both its recall on normal images and its Attack Success Rate — how reliably the trigger fools it.
## Requirements
Google Colab, plus torch, torchvision (with pretrained ResNet18 weights), pandas, PIL, and scikit-learn.
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing train.zip and test.zip.
## Output
**Setup and data loading:** directory listings confirm the extracted train/test folders and their structure. The labels CSV is loaded and previewed to confirm the columns are as expected.

**Model setup:** the full ResNet18 architecture is printed once instantiated, confirming the final layer was correctly swapped for a single-output binary classifier.

**Initial training (clean data):** each of the 5 epochs prints its average loss, e.g. Epoch [1/5], Loss: 0.4213, tracking how well the model learns to detect pedestrians. After training, a single Test Accuracy value is printed on the untouched test set — this is the baseline performance before anything else happens.

**Temperature scaling:** raw logits and true labels are collected across the whole test set, and their shapes are printed as a sanity check. The model's probabilities under temperature scaling are also visualized. Then, for each of the three temperature values (0.5, 1.0, 2.0), a line is printed showing the resulting accuracy — for example T = 0.5, Accuracy = 0.8123. Comparing these three numbers shows whether reshaping the confidence scores (without changing the underlying predictions much) helps or hurts overall correctness, and a plot visualizes the calibrated probability distribution.

**Backdoor attack setup:** after poisoning 10% of pedestrian-positive training images with a small red square (and flipping their label), the number of poisoned samples is printed, confirming exactly how many images were altered.

**Backdoor fine-tuning:** the model is fine-tuned for 3 more epochs on this poisoned dataset, with each epoch printing its own loss line, e.g. Backdoor Epoch [1/3], Loss: 0.2876 — separate from the earlier clean-training loss so the two phases are easy to distinguish.

**Final backdoor evaluation:** two key numbers are printed at the end. Clean Recall reports how well the now-backdoored model still detects pedestrians in normal, untriggered images — ideally close to its original performance, meaning the backdoor is well hidden. The Attack Success Rate (ASR) reports what fraction of triggered pedestrian images successfully fool the model into predicting "no pedestrian." A high ASR alongside a high Clean Recall is the sign of a successful, stealthy backdoor: the model behaves normally almost all the time, but fails predictably and dangerously whenever the trigger is present.

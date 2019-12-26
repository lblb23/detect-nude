# detect_porn
Draft project for nudity detection with simple convolutional neural network in Keras

## Getting Started

1. Clone repo
2. Choose class mode in congif.yaml and set path to model
3. Run model on any image:
```
python predict.py --img_path image.jpg
```
4. You can evaluate model on dataset:
```
python eval.py
```
5. You can train model on dataset:
```
python train.py
```

## Dataset

If you need the dataset, please send to me message on buliginleo @ yandex.ru

### Results

| Class_mode  | Accuracy |
| ------------- | ------------- |
| 2 classes (erotic_and porn, legal)  | 0.8946  |
| 3 classes (erotic, porn, legal) | 0.7300 |

### Classes

Legal - images without naked people
Erotic - images with half-naked people (for example, in swimsuits)
Porn - images with completely naked people or sex scenes

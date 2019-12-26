import argparse

import cv2
import numpy as np
import yaml
from keras.models import load_model

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',
                    default='data_3classes/validation/legal/VYZGTF26GW.jpg',
                    dest='img_path',
                    help='path to image')

args = parser.parse_args()
filepath = args.img_path

model = load_model(config['path_to_trained_model'])

img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.reshape(img, [1, 224, 224, 3])

classes = model.predict(img)

if config['class_mode'] == '2classes':
    print('Probability of legal:')
elif config['class_mode'] == '3classes':
    print('Class indices: 0 - erotic, 1 - legal, 2 - porn')

print(classes)

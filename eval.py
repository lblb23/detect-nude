import yaml
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1. / 255)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config['class_mode'] == '2classes':
    class_mode = 'binary'
elif config['class_mode'] == '3classes':
    class_mode = 'categorical'

validation_generator = test_datagen.flow_from_directory(
    f"data_{config['class_mode']}/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode=class_mode)

model = load_model(config['path_to_trained_model'])
print(model.evaluate_generator(validation_generator, verbose=1))

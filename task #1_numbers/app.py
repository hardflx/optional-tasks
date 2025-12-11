import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model_path = r"C:/Users/hardf/Desktop/otional tasks/task #1_numbers/model/mnist-model.h5"
model = load_model(model_path, compile=False)

images_folder = r"C:/Users/hardf/Desktop/otional tasks/task #1_numbers/data for task"

def prepare_image(path, target_size=(28, 28)):
    img = Image.open(path).convert("L")
    img = img.resize(target_size)
    img = np.array(img).astype("float32") / 255
    img = img.reshape(1, target_size[0], target_size[1], 1)
    return img

counts = [0] * 10

for filename in os.listdir(images_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(images_folder, filename)
        img = prepare_image(img_path)
        pred = model.predict(img, verbose=0)
        digit = int(np.argmax(pred))
        counts[digit] += 1

counts = [f'{i}_{count}' for i, count in enumerate(counts)]
print(counts)
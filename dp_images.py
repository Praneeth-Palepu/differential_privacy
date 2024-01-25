import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Flatten
from keras.utils import load_img, img_to_array
from keras.applications import imagenet_utils, VGG16
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

class ImagePrivacy:
    def __init__(self, target_size=(224, 224)):
        self.target_size= target_size

    def apply_differential_privacy(self, image_array, epsilon=0.01):
        np.random.seed(0)
        noise = np.random.laplace(0, scale=1/epsilon, size=image_array.shape)
        image_array_with_dp = image_array + noise
        return image_array_with_dp

    def load_image_dataset(self, dataset_path, allowed_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'], test_split_size=0.2, random_state=0):
        images, labels = [], []
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                for file in tqdm(os.listdir(folder_path)):
                    imgPath = os.path.join(folder_path, file)
                    file, extension = os.path.splitext(imgPath)
                    if extension.lower() in allowed_extensions:
                        img = load_img(imgPath, target_size=self.target_size)
                        img_arr = img_to_array(img)
                        img_arr = imagenet_utils.preprocess_input(img_arr)
                        images.append(img_arr)
                        labels.append(folder)
        images = np.array(images)
        labels = np.array(labels)

        xtr, xte, ytr, yte = train_test_split(images, labels, test_size=test_split_size, random_state=random_state)
        return (xtr, ytr), (xte, yte)

    def feature_extraction_with_vgg16(self, image_array):
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        model = Sequential([
            vgg16,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features
    
    def train(self, train_image_features, target_values):
        lr = LogisticRegression(solver= 'liblinear')
        lr.fit(train_image_features, target_values)
        return lr
    
    def test(self, model, test_image_features, ground_truth):
        prediction = model.predict(test_image_features)
        accuracy = accuracy_score(ground_truth, prediction)
        return accuracy
    



import numpy as np
from keras.utils import load_img, img_to_array, array_to_img
from keras.applications import VGG16, imagenet_utils
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score 
from keras.models import Sequential
from keras.layers import Flatten
import os
from tqdm import tqdm

class ImagePrivacy:
    def __init__(self, img_target_size=(224, 224)):
        self.target_size = img_target_size
    
    def load_dataset(self, image_path, folders_to_exclude_lst= [], valid_img_file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'], test_size=0.2, random_state=0):
        images, labels = [], []
        if os.path.isdir(image_path):
            for folder in os.listdir(image_path):
                if not folder in folders_to_exclude_lst:
                    folder_path = os.path.join(image_path, folder)
                    if os.path.isdir(folder_path):
                        for file in tqdm(os.listdir(folder_path)):
                            file_path = os.path.join(folder_path, file)
                            file, extension = os.path.splitext(file_path)
                            if extension in valid_img_file_extensions:
                                img = load_img(file_path, target_size= self.target_size)
                                img_arr = img_to_array(img)
                                img_arr = imagenet_utils.preprocess_input(img_arr)
                                images.append(img_arr)
                                labels.append(folder)
        images = np.array(images)
        labels = np.array(labels)
        xtr, xte, ytr, yte = train_test_split(images, labels, test_size= test_size, random_state= random_state)
        del images
        del labels
        return (xtr, ytr), (xte, yte)
    
    def apply_privacy_to_image(self, image_array, epsilon=0.01, clip= True):
        scale = 1/epsilon
        np.random.seed(0)
        noise = np.random.laplace(0, scale, size= np.array(np.hstack((self.target_size, np.array([3])))))
        privacy_applied_image_array = image_array + noise
        if clip:
            privacy_applied_image_array = np.clip(privacy_applied_image_array, 0, 255)
        return privacy_applied_image_array
    
    def extract_features(self, image_array, feature_extractor_model):
        model = Sequential([
            feature_extractor_model,
            Flatten()
        ])
        features = model.predict(image_array)
        return features
    
    def train(self, model, image_features, target_variable):
        model.fit(image_features, target_variable)
        return model
    
    def test(self, trained_model, test_image_features, ground_truth):
        prediction = trained_model.predict(test_image_features)
        accuracy = accuracy_score(ground_truth, prediction)
        return round(accuracy *100, 2)
    
    def run_pipeline(self, train_image_array, epsilon, clip, target_variable, test_image_array, ground_truth, fem_dp, fem_or, dp_model, or_model, feature_reduction_bool= False, n_components=None):
        tr_img_arr_dp = self.apply_privacy_to_image(image_array= train_image_array, epsilon= epsilon, clip = clip)
        tr_features_dp = self.extract_features(image_array = tr_img_arr_dp, feature_extractor_model= fem_dp)
        tr_features_or = self.extract_features(image_array = train_image_array, feature_extractor_model= fem_or)
        te_features_or = self.extract_features(image_array= test_image_array, feature_extractor_model= fem_or)
        if feature_reduction_bool:
            svd_dp = None
            if n_components is None:
                svd_dp = TruncatedSVD(n_components=1000, random_state=0)
                svd_or = TruncatedSVD(n_components=1000, random_state=0)
            else:
                svd_dp = TruncatedSVD(n_components=n_components, random_state=0)
                svd_or = TruncatedSVD(n_components=n_components, random_state=0)
            
            if svd_dp is not None:
                tr_features_dp = svd_dp.fit_transform(tr_features_dp)
            
            if svd_or is not None:
                tr_features_or = svd_or.fit_transform(tr_features_or)
                te_features_or = svd_or.transform(te_features_or)

        dp_model = self.train(dp_model, image_features= tr_features_dp, target_variable= target_variable)
        or_model = self.train(or_model, image_features= tr_features_or, target_variable= target_variable)
        dp_acc = self.test(trained_model = dp_model, test_image_features= te_features_or, ground_truth= ground_truth)
        or_acc = self.test(trained_model = or_model, test_image_features= te_features_or, ground_truth= ground_truth)

        return (or_acc, dp_acc)

        





from dp_images import ImagePrivacy
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

print("\nInitializing...")
ip = ImagePrivacy()

print("\nLoading Image dataset...")
(xtr, ytr), (xte, yte) = ip.load_image_dataset(r'/Users/praneethkumarpalepu/Documents/machine_learning/august_batch/animals/images')

epsilon_range = [0.01, 0.1, 1, 10]

epsilon_acc_dict = {'privacy_budget':[], 'accuracy_on_test_data':[]}

for epsilon in epsilon_range:
    print("\nApplying differential privacy to training images at source")
    xtr_dp = ip.apply_differential_privacy(xtr, epsilon= epsilon)

    print("\nExtracting features from training images...")
    train_features = ip.feature_extraction_with_vgg16(xtr)
    #joblib.dump(train_features, "train_features.joblib")
    print("\nExtracting features from Differential Privacy applied training images...")
    train_features_with_dp = ip.feature_extraction_with_vgg16(xtr_dp)
    #joblib.dump(train_features_with_dp, "train_features_with_dp.joblib")
    print("\nExtracting features from test images...")
    test_features = ip.feature_extraction_with_vgg16(xte)
    #joblib.dump(test_features, "test_features.joblib")

    '''
    train_features = joblib.load("train_features.joblib")
    train_features_with_dp = joblib.load("train_features_with_dp.joblib")
    test_features = joblib.load("test_features.joblib")
    '''

    print("\nTraining logistic regression model on original data...")
    model_without_dp = ip.train(train_features, ytr)
    print("\nTraining logistic regression model on Differential Privacy applied data...")
    model_with_dp = ip.train(train_features_with_dp, ytr)

    print("\nCalculating accuracy of logistic regression model on original data...")
    model_without_dp_acc = ip.test(model_without_dp, test_features, yte)
    print("\nCalculating accuracy of logistic regression model on differential privacy applied data...")
    model_with_dp_acc = ip.test(model_with_dp, test_features, yte)

    print(f"\nAccuracy of model on DP trained data: {round(model_with_dp_acc*100, 2)}%")
    print(f"\nAccuracy of model on Original data: {round(model_without_dp_acc*100, 2)}%")

    epsilon_acc_dict['privacy_budget'].append(epsilon)
    epsilon_acc_dict['accuracy_on_test_data'].append(model_with_dp_acc)

df = pd.DataFrame(epsilon_acc_dict)
print(df)
import streamlit as st
from PIL import Image
from privacy import ImagePrivacy
from sklearn.linear_model import LogisticRegression
from keras.applications import VGG16
from keras.utils import array_to_img
import numpy as np

img_icon = Image.open(r'icon.png')
ip = ImagePrivacy()

st.set_page_config(layout = 'wide', page_icon = img_icon)
st.title(":orange[Differential Privacy]")

folder_path = st.sidebar.text_input("Enter the folder path", key="folder_input")

if folder_path != '':
    (xtr, ytr), (xte, yte) = ip.load_dataset(image_path= folder_path)
    image_preview, accuracy_privacy = st.tabs([":blue[Image Preview]", ":blue[Accuracy with Privacy]"])
    with image_preview:
        privacy_budget_min = st.number_input(":blue[Enter the min range of privacy budget]", min_value= 0.0001, max_value=1.0, value= 0.1, format = "%.4f")
        privacy_budget_max = st.number_input(":blue[Enter the max range of privacy budget]", min_value= 0.01, max_value=100.0, value= 0.1, format = "%.4f")
        intervals = st.number_input(":blue[Choose the number of intervals between min and max privacy budget]", min_value= 3, max_value=10, value= 3)
        image_index = st.number_input(":blue[Enter an index for image preview]", min_value = 1)
        columns = st.columns(intervals+1)
        if folder_path != '':
            column_counter = 0
            columns[column_counter].image(array_to_img(xtr[image_index-1]), caption= "Image with No Dp", use_column_width = True)
            for epsilon in np.linspace(start = privacy_budget_max, stop = privacy_budget_min, num = intervals):
                column_counter+=1
                xtr_dp = ip.apply_privacy_to_image(xtr, epsilon= epsilon)
                columns[column_counter].image(array_to_img(xtr_dp[image_index-1]), caption = f"Image with {round(epsilon, 3)} dp", use_column_width = True)
    with accuracy_privacy:
        privacy_budget = st.number_input(":blue[Select your privacy budget]", min_value= 0.001, max_value=5.0)
        calc_acc = st.button(":blue[Calculate accuracy]")
        if calc_acc:
            fem_dp = VGG16(include_top="False", weights= 'imagenet', input_shape=(224, 224, 3))
            fem_or = VGG16(include_top="False", weights= 'imagenet', input_shape=(224, 224, 3))
            dp_lr = LogisticRegression(solver= 'liblinear')
            or_lr = LogisticRegression(solver= 'liblinear')
            result = ip.run_pipeline(train_image_array= xtr, epsilon= privacy_budget, clip = True, target_variable= ytr, test_image_array = xte, ground_truth = yte, fem_dp = fem_dp, fem_or = fem_or, dp_model = dp_lr, or_model = or_lr)
            st.write(f":blue[Accuracy of model trained on original data on test data is:] :green[{result[0]}%]")
            st.write(f":blue[Accuracy of model trained on DP applied data on test data is:] :green[{result[1]}%]")



import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("model.h5")

### load file
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg", "tiff", "webp"])
st.write("do not reli on these predections 100% chec a doctor")

map_dict = {0: 'Covid',
            1: 'Normal'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(512,512))
    # Now do something with the image! For example, let's display it:
    st.image(cv2.resize(opencv_image,(200,200)), channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("predict label for this X-ray")    
    if Genrate_pred:

        prediction = model.predict(img_reshape).argmax()
	
        st.title("this is a {} chest x-ray and i am sure of it : {}%".format(map_dict [prediction], round(model.predict(img_reshape).max()*100, 6)))
	

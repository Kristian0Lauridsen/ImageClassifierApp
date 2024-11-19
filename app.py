#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from fastai.vision.core import PILImage
from fastai.learner import load_learner
from PIL import ImageOps
from io import BytesIO


# In[8]:


name_mapping = {
    'cavallo': 'horse',
    'pecora': 'sheep',
    'elefante': 'elephant',
    'gatto': 'cat',
    'scoiattolo': 'squirrel',
    'gallina': 'chicken',
    'ragno': 'spider',
    'mucca': 'cow',
    'cane': 'dog',
    'farfalla': 'butterfly'
}


# In[3]:


import pathlib

EXPORT_PATH = pathlib.Path("model3.pkl")  # Define the path to your model

learn_inf = load_learner(EXPORT_PATH)  # Load the learner


# In[9]:


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize img as None to avoid unnecessary initial prediction
img = None

if uploaded_file:
    try:
        # Read and process the uploaded image
        file_bytes = BytesIO(uploaded_file.read())
        img = PILImage.create(file_bytes)
        img = ImageOps.fit(img, (224, 224))  # Resize if necessary
        st.image(img.to_thumb(256, 256), caption="Uploaded Image")

        # Make prediction
        pred_class, pred_idx, probs = learn_inf.predict(img)
        # Map the predicted class to its English label
        english_label = name_mapping.get(pred_class, pred_class)  # Fallback to original if not in mapping
        st.write(f"Prediction: {english_label}")
        st.write(f"Confidence: {probs[pred_idx]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning("Please upload an image to classify.")


# In[ ]:





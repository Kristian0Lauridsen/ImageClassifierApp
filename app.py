#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.widgets import *
from types import SimpleNamespace
from fastai.vision.core import PILImage
from fastai.learner import load_learner
import streamlit as st
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


from contextlib import contextmanager
import pathlib

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


# In[4]:


EXPORT_PATH = pathlib.Path("model3.pkl")

with set_posix_windows():
    learn_inf = load_learner(EXPORT_PATH)


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





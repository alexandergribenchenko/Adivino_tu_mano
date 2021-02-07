# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 2021
@author: Alexander Ortega
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def transform_and_predict(image, model):
        size = (64,64)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_paso_00 = np.asarray(image)
        image_paso_01 = image_paso_00[:,:,0]
        image_paso_02 = np.array([image_paso_01])
        image_paso_03= image_paso_02.reshape(image_paso_02.shape[0], image_paso_02.shape[1], image_paso_02.shape[2], 1)
        image_paso_04 = image_paso_03.astype('float32')/255
        prediction = model.predict(image_paso_04)
        return prediction

model = tf.keras.models.load_model('Adivino_tu_mano_(Modelo).hdf5')

st.write("""
         # !Adivino tu mano!
         """
         )

st.write("Carga una imagen en formato jpg o png en la que hagas el gesto de un número entre 0 y 5 con una única mano.")

st.write("¡A que soy capaz de adivinarlo!")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("Aún no has subido ninguna imagen. !Anímate a hacerlo!")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = transform_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("!Yo que te digo que eso es un cero!")
    elif np.argmax(prediction) == 1:
        st.write("!Yo que te digo que eso es un uno!")
    elif np.argmax(prediction) == 2:
        st.write("!Yo que te digo que eso es un dos!")
    elif np.argmax(prediction) == 3:
        st.write("!Yo que te digo que eso es un tres!")
    elif np.argmax(prediction) == 4:
        st.write("!Yo que te digo que eso es un cuatro!")
    elif np.argmax(prediction) == 5:
        st.write("!Yo que te digo que eso es un cinco!")
    
    st.text("Aunque a veces me la pones difícil, creo que esta es la probabilidad de que pueda \nser cada uno de los 6 números (0: cero, 1: uno, 2: dos, 3: tres, 4: cuatro, 5: cinco): ")
    st.write(prediction)

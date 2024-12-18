import streamlit as st
import tensorflow as tf
import numpy as np
import keras.utils as image
import matplotlib.pyplot as plt

# Carregando o modelo treinado
model = tf.keras.models.load_model('modelo_fogo.h5')

# Função para processar a imagem recebida e realizar a previsão
def predict_fire(img):
    img = img.resize((224, 224))  
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Fazendo a previsão
    prediction = model.predict(img_array)
    return 'Sem fogo detectado' if prediction[0] > 0.3 else 'Fogo detectado'

# Criando a interface do Streamlit
st.title("Detecção de Fogo em Imagens")
st.write("Envie uma imagem para verificar se há fogo ou não.")

# Permite ao usuário fazer upload de uma imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png"])

# Se o usuário enviar uma imagem
if uploaded_file is not None:
    
    img = image.load_img(uploaded_file)
    st.image(img, caption="Imagem carregada", use_container_width=True)

    
    result = predict_fire(img)
    st.write(f"Resultado da previsão: {result}")

    
    st.write("Confiança da previsão:")
    confidence = model.predict(np.expand_dims(np.array(img.resize((224, 224)))/255.0, axis=0))
    st.bar_chart(confidence[0])  # Exibindo gráfico de confiança

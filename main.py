from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import joblib
import tensorflow as tf

# Load model and tokenizer
MODEL_WEIGHTS = "Encoder_Decoder_global_attention.h5"
TOKENIZER_PATH = "tokenizer.pkl"
model, tokenizer = None, None

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model, tokenizer = create_model()

def create_model():
    tokenizer = joblib.load(TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index)
    embedding_dim = 300
    dense_dim = 512
    dropout_rate = 0.2
    max_pad = 29

    image1 = tf.keras.Input(shape=(224,224,3))
    image2 = tf.keras.Input(shape=(224,224,3))
    caption = tf.keras.Input(shape=(max_pad,))

    encoder_output = encoder(image1, image2, dense_dim, dropout_rate)
    output = Decoder(max_pad, embedding_dim, dense_dim, 100, vocab_size)(encoder_output, caption)
    
    model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
    model.load_weights(MODEL_WEIGHTS)
    return model, tokenizer

@app.post("/predict/")
async def predict(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    image1 = cv2.imdecode(np.frombuffer(image1.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image2 = cv2.imdecode(np.frombuffer(image2.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    if image1 is None or image2 is None:
        return JSONResponse(content={"error": "Error loading images."})

    caption = predict1(image1, image2, [model, tokenizer])
    return JSONResponse(content={"caption": caption})

def predict1(image1, image2, model_tokenizer):
    model, tokenizer = model_tokenizer
    return greedy_search_predict(image1, image2, model, tokenizer)

def greedy_search_predict(image1, image2, model, tokenizer):
    image1 = tf.image.resize(image1, (224, 224))
    image2 = tf.image.resize(image2, (224, 224))
    
    encoder_output = model.get_layer('image_encoder')(image1)
    decoder_h = tf.zeros_like(encoder_output[:,0])

    generated_text = []
    max_pad = 29
    for i in range(max_pad):
        if i == 0:
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))
        output, decoder_h, _ = model.get_layer('decoder').onestepdecoder(caption, encoder_output, decoder_h)
        max_prob = tf.argmax(output, axis=-1)
        caption = np.array([max_prob])
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break
        else:
            generated_text.append(tf.squeeze(max_prob).numpy())

    return tokenizer.sequences_to_texts([generated_text])[0]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

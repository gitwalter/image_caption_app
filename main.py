import streamlit as st
import torch
from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

@st.cache_resource
def load_blip2_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"        
    processor = Blip2Processor.from_pretrained(model_key)
    # by default `from_pretrained` loads the weights in float32
    # we load in float instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained(model_key, torch_dtype=torch.float)
    model.to(device)
    return device, model, processor
    

# Define models and their corresponding names
models = {
    "ydshieh/vit-gpt2-coco-en": "vit-gpt2-coco-en",
    "Salesforce/blip2-opt-2.7b": "blip2-opt-2.7b"
}

# Dropdown for selecting the model
selected_model = st.selectbox("Select Model", options=list(models.values()))

# Get the model key based on the selected model name
model_key = [key for key, value in models.items() if value == selected_model][0]

upload_image_name = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if upload_image_name is not None:
    uploaded_image = Image.open(upload_image_name)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    
question = st.text_input('Question', 'Ask the model about the picture')

if st.button("Generate Caption"):
    if selected_model == "vit-gpt2-coco-en":
        # Initialize the caption pipeline with the selected model
        caption = pipeline('image-to-text', model=model_key)
        captions = caption(uploaded_image)
        model_answer = captions[0]['generated_text']
    else:       
        device, model, processor = load_blip2_model()        
       
        if question:
            inputs = processor(uploaded_image, question, return_tensors="pt").to(device, torch.float)
            out = model.generate(**inputs, max_new_tokens=20)
            model_answer = processor.decode(out[0], skip_special_tokens=True).strip()
        else:
            inputs = processor(uploaded_image, return_tensors="pt").to(device, torch.float)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            model_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    st.write(model_answer)
    

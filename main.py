import base64
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil
import subprocess
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()



#STYLING ...

# Convert Background Image to Base64 for Streamlit Styling ...
def get_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("Background.jpg")
page_bg = f"""
<style>
/* Background for main app */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    filter: brightness(0.8);
}}

/* Header transparent */
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

/* Text Styles */
h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* লেখা পরিষ্কার বোঝার জন্য শ্যাডো */
}}

/* File Uploader Design */
[data-testid="stFileUploaderDropzone"] {{
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 2px dashed #ffffff !important;
    border-radius: 10px;
}}

/* Button Design */
div.stButton > button {{
    background-color: #FF5733 !important;
    color: white !important;
    border-radius: 20px !important;
    width: 100%;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Permanent Notice for Users ...
st.markdown("""
<div style="background-color: rgba(255, 165, 0, 0.2); padding: 15px; border-radius: 10px; border-left: 5px solid #2a7eb1; margin-bottom: 20px;">
    <p style="color: white; margin: 0; font-weight: 500;">
        ⚠️ <b>Note:</b> This system supports only <b>single-user</b> access at a time and use only <b>one file</b> at a time.
    </p>
</div>
""", unsafe_allow_html=True)




# API KEY HANDLING ...

# API Key loading ...
API_KEY = os.getenv("API_KEY_OPENAI")
client = OpenAI(api_key= API_KEY)



# FUNCTIONS DESIGN ...

# Video to mp3 conversation function ...
def video_to_mp3(video_path):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join("Uploads" , f"{filename}.mp3")
    subprocess.run(["ffmpeg" , "-i" , video_path , audio_path] , check= True)
    return audio_path

# mp3 to chunks_json conversation function ...
def audio_to_text(audio_path):
    with open(audio_path , "rb") as f:
        response = client.audio.transcriptions.create(
            model= "gpt-4o-transcribe",
            file= f
        )
    transcripted_text = response.text
    if not transcripted_text:
        return None
    chunks = []
    words = transcripted_text.split()
    chunk_size = 50
    overlap = 10
    step = chunk_size - overlap
    for i in range(0 , len(words) , step):
        chunk = " ".join(words[i : i+chunk_size])
        if chunk:
            chunks.append(chunk)
    list_of_dict_chunks = [{"text" : chunk} for chunk in chunks]
    chunks_json = {"proper_chunks" : list_of_dict_chunks}
    os.makedirs("jsons" , exist_ok= True)
    with open("jsons/chunks.json" , "w" , encoding='utf-8') as f:
        json.dump(chunks_json , f , indent = 2 , ensure_ascii=False)
    json_path = "jsons/chunks.json"
    return json_path

# PDF to chunks_json conversation function ...
def pdf_to_chunks_json(pdf_path):

    with open(pdf_path, "rb") as f:
        file = client.files.create(
            file=f, 
            purpose="assistants"
        )

    response = client.responses.create(
        model="gpt-4.1",
        input=[
           {
            "role": "user", "content": [{"type": "input_text", "text": "Extract all text from this PDF clearly."},
                {
                    "type": "input_file",
                    "file_id": file.id
                }
                                       ]
            }
               ]
    )

    extracted_text = response.output_text

    if not extracted_text:
        return None

    words = extracted_text.split()
    chunks = []

    chunk_size = 100
    overlap = 20
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    list_of_dict_chunks = [{"text": chunk} for chunk in chunks]
    chunks_json = {"proper_chunks": list_of_dict_chunks}

    os.makedirs("jsons", exist_ok=True)
    filename = os.path.basename(pdf_path).replace(".pdf", "")
    json_path = f"jsons/{filename}_chunks.json"

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(chunks_json, f, indent=2, ensure_ascii=False)

    return json_path

# Image to chunks_json conversation function ...
def image_to_chunks_json(image_path):
   
    with open(image_path, "rb") as f:
        file = client.files.create(
            file=f, 
            purpose="assistants"
        )

    response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract all text from this image clearly."},
                {
                    "type": "input_image",
                    "file_id": file.id
                }
                       ]
        }
          ]
    )

    extracted_text = response.output_text

    if not extracted_text:
        return None

    words = extracted_text.split()
    chunks = []

    chunk_size = 80
    overlap = 15
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    list_of_dict_chunks = [{"text": chunk} for chunk in chunks]
    chunks_json = {"proper_chunks": list_of_dict_chunks}

    os.makedirs("jsons", exist_ok=True)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    json_path = f"jsons/{filename}_chunks.json"

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(chunks_json, f, indent=2 , ensure_ascii=False)

    return json_path

# Text to embedding function ...
def create_embedding(json_path):
    with open(json_path , "r" ,encoding='utf-8') as f:
        content = json.load(f)
    text_list = [c["text"] for c in content["proper_chunks"]]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    embedding = [item.embedding for item in response.data]
    for i , chunk in enumerate(content["proper_chunks"]):
        chunk["embedding"] = embedding[i]
        chunk["chunk_id"] = i
    df = pd.DataFrame.from_records(content["proper_chunks"])
    return df

# Cosine similarity function ...
def find_similar_chunks(incoming_query , df):
    response = client.embeddings.create(
        model= "text-embedding-3-small",
        input= [incoming_query]
    )
    question_embedding = [item.embedding for item in response.data]
    similarities = cosine_similarity(np.vstack(df["embedding"]), question_embedding).flatten()
    top_results = 3
    max_index = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_index]

    prompt = f"""
    Here are some chunks which have included text:
    {new_df[["text"]].to_json(orient = "records")}
    --------------------------------------------------------
    Based on the provided chunks answer the following quesion(at the time of answer the question don't mention the format it is just for you),
    and answer like a human, you will give the answer in standard format and don't mention the format it is for you,
    and dont give point by point answer , give answer in paragraph,and remember you don't have to lot of things give just what is required and perfect, 
    and if you don't understand the content thend directly say i am not understanding ur QS and if the answer is not in the content then say i am unable to find ur qs's answer:
    {incoming_query}
    """

    response = client.responses.create(
        model='gpt-5-mini',  
        input= prompt
    )
    content = response.output_text
    return content

# Remove Directory function after processing ...
def remove_upload_dir():
    if os.path.exists("Uploads"):
        shutil.rmtree("Uploads")

def remove_jsons_dir():
    if os.path.exists("jsons"):
        shutil.rmtree("jsons")



# THE STREAMLIT APP STARTS FROM HERE ...

st.set_page_config(page_title="RAG BASED AI TEACHING ASSISTANT", layout="centered")
st.title("📄 Multi-Model RAG App")

uploaded_file = st.file_uploader(
    "Upload File (PDF / Image / Video / Audio)",
    type= ["pdf" , "png" , "jpg" , "jpeg" , "mp3" , "mp4"]
)

if uploaded_file:

    if uploaded_file.size is not None:
        if uploaded_file.size > 75 * 1024 * 1024:
            st.error("❌ File too large! Please upload under 75MB.")
        else:
            st.success(f"File uploaded : {uploaded_file.name}")
            
            if st.button("Proceed File Processing"):

                remove_upload_dir()
                remove_jsons_dir()

                if "json_path" in st.session_state:
                    del st.session_state["json_path"]
                st.info("Please wait while we process your file ...")
                os.makedirs("Uploads" , exist_ok= True)
                file_path = f"Uploads/{uploaded_file.name}"
                with open (file_path , "wb") as f:
                    f.write(uploaded_file.read())
                ext = uploaded_file.name.split(".")[-1].lower()

                if ext == "pdf":
                    json_path = pdf_to_chunks_json(file_path)
                elif ext in ["png" , "jpg" , "jpeg"]:
                    json_path = image_to_chunks_json(file_path)
                elif ext == "mp3":
                    json_path = audio_to_text(file_path)
                elif ext == "mp4":
                    audio_path = video_to_mp3(file_path)
                    json_path = audio_to_text(audio_path)
                else:
                    st.error("Sorry, this file can't be processed.")
                    remove_upload_dir()
                    remove_jsons_dir()
                    st.stop()
                if json_path is None:
                    st.error("Sorry, we couldn't extract any text from the uploaded file.")
                    remove_upload_dir()
                    remove_jsons_dir()
                    st.stop()

                st.session_state["json_path"] = json_path
                st.success("File processed successfully! You can now ask questions related to the content")
            
            if "json_path" in st.session_state:
                incoming_query = st.text_input("Ask a Question related to the content:")
                if st.button("Search"):
                    st.info("Finding the answer to your question ...")
                    if not incoming_query.strip():
                        st.warning("Please enter a question before searching")
                    else:
                        df = create_embedding(st.session_state["json_path"])
                        answer = find_similar_chunks(incoming_query , df)
                        st.subheader("Answer :")
                        st.write(answer)
                        
                                

                

import os
import sys
import logging
import subprocess
import tempfile
import soundfile as sf
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from schemas import ChatResponse
from callback import StreamingLLMCallbackHandler
from websockets.exceptions import ConnectionClosedOK
from query_data import get_chain, call_chain
import faiss
# import whisper
# from whisper import load_model
from langchain.llms import OpenAI
# from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile,Form
import requests
from typing_extensions import Annotated
import json
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from config import TTS_ENABLED
from langchain.callbacks import get_openai_callback

# To load the environment variables from the .env file
load_dotenv()

# set environmnt variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

class QueryRequest(BaseModel):
    query: str
    
# whisper_model = whisper.load_model("base.en")
command = ['ffmpeg', '-i', '-', '-f', 'wav', '-']

# index = faiss.read_index("D:\Downloads\adiray_bot_f\faiss\index.fais")
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.mount(
    "/templates",
    StaticFiles(directory=os.path.dirname(os.path.realpath(__file__)) +"/templates"),
    name="static",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# templates = Jinja2Templates(directory="templates")
# define FAISS vectorstore and store embedding into local
db = FAISS.load_local(
        folder_path="faiss",
        index_name="index",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        # allow_dangerous_deserialization=True,
    )


@app.post("/audio")
async def audio(file: Annotated[bytes, File()]):# Annotated[bytes, File()]):
   
    """
    Audio API endpoint
    """
    print("Started processing audio")
    
    with open('output_file.wav',mode='wb') as f:
                
        f.write(file)

    audio_file = open("output_file.wav", "rb")

    # TTS via whisper
    # resp = whisper_model.transcribe(output_file_path)
    # text = resp['text']

    # TTS via OpenAI whisper
    text = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

    message = {"response": text}
    return JSONResponse(message)
    
@app.get("/req")
async def index(request: Request):
    return print("hi")




#chain is defined without streaming
qa_chain_wo_streaming = call_chain()

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    chat API endpoint for tes  with get_openai_callback() as cb:
         print(cb)ting
    """
    text = request.query
    # max_context_length = 4097 - 256
    print("chat works")
    relevant_docs = db.similarity_search(text)
    print(relevant_docs)
    
    result = {
        "output":qa_chain_wo_streaming(
        {"input_documents": relevant_docs, "human_input": text, "chat_history": ""},
        return_only_outputs=True
        )["output_text"]
    }
    return JSONResponse(content=result)


#applied web scokets 
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        os.remove('static/tts.mp3')
    except:
        pass
    
    stream_handler = StreamingLLMCallbackHandler(websocket)
    qa_chain = get_chain(stream_handler)

    while True:
        try:
            # Receive and send back the client message
            user_msg = await websocket.receive_text()
            resp = ChatResponse(sender="human", message=user_msg, type="stream", integer=0)
            await websocket.send_json(resp.model_dump())

            # To construct a streaming response
            start_resp = ChatResponse(sender="bot", message="", type="start", integer=0)
            await websocket.send_json(start_resp.model_dump())
            
            # relevant_docs = get_relevant_documents(user_msg)
            relevant_docs=db.similarity_search(user_msg, k=1)
            
            #result will store response geenrated by chain
            with get_openai_callback() as cb:
                    
                result = await qa_chain.acall(
                {"input_documents": relevant_docs, "human_input": user_msg}, 
                return_only_outputs=True
                )
          
                output = result["output_text"] 
                print(cb)
            
            end_resp = ChatResponse(sender="bot", message="", type="end", integer=0)
            await websocket.send_json(end_resp.model_dump())
            
            # @sample_data_start
            # sample_response = ChatResponse(sender="bot", message="Testing in progress", type="stream", integer=0)
            # await websocket.send_json(sample_response.dict())
            # output = 'Testing in progress'
            # audio_bytes_file = open('tts.txt',mode='rb')
            # data = audio_bytes_file.read()
            # with open('tts_sample.mp3',mode='wb') as f:
                
            #     f.write(data)
            # data_list = list(data)
            # print(data_list[:10], type(data_list[0]), len(data_list))
            # @sample_data_end
            
            
            # @tts_start ++++++++++++++
            
            if TTS_ENABLED:
                response = client.audio.speech.create(
                            model="tts-1",
                            voice="shimmer",
                            input=output
                        )
                response.stream_to_file('static/tts'+".mp3")

                # Send audio in bytes at end
                # await websocket.send_bytes(data)
                
                audio_start = ChatResponse(sender="bot", message='',type="audio_start", integer=0)
                await websocket.send_json(audio_start.model_dump())
                
                for data in response.iter_bytes(None):
                    await websocket.send_bytes(data)
                    
                audio_end = ChatResponse(sender="bot", message='',type="audio_end", integer=0)
                await websocket.send_json(audio_end.model_dump())
            
            # @tts_end ++++++++++++++
            
        except WebSocketDisconnect:
            logging.info("WebSocketDisconnect")
            # To try to reconnect with back-off
            break
        except ConnectionClosedOK:
            logging.info("ConnectionClosedOK")
            # To handle this
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
                integer=0
            )
            await websocket.send_json(resp.model_dump())

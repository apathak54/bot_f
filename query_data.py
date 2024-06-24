
import os
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

# To load the environment variables from the .env file
load_dotenv()

# Access the variables using os.environ
openai_api_key = os.environ.get("OPENAI_API_KEY")




prompt = PromptTemplate(
    input_variables=["chat_history", "context", "human_input"], 
    template="""You are an AI-powered Trade chatbot designed to assist Merchant Traders.
    give the answer based on given documents and human resources-related queries.
    {chat_history}
    {context}
    Question: {human_input}
    Chatbot:"""
    )


def get_chain(stream_handler, tracing: bool = False) -> ConversationChain:
    """Create a ConversationChain for question/answering.
    """

    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    streaming_llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callback_manager=stream_manager,
        temperature=0,
        # verbose=True,
        # model_kwargs={"top_p": 0.2},
    )

	#added conversation buffer window memory for storing preious three conversations we had with the chatbot
    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=3)
  
	#used load qa chain for question answering
    qa = load_qa_chain(streaming_llm, callback_manager=manager, chain_type="stuff", memory=memory, prompt=prompt)
       
    return qa

def call_chain():
    """
    calling chain without streaming
    """

    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=3)
    qna = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
    return qna


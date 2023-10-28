import logging
import sys
sys.path.append("../")


import chainlit as cl
import pandas as pd
from chainlit.input_widget import Select
from transformers import AutoTokenizer, AutoModel


from chat.constants import CHAT_MODES, EMBEDDING_MODEL_DEVICE, EMBEDDING_MODEL_REPO_NAME, LLMS, PROCESSED_DATA_FILE, DATA_FOLDER
from chat.utils import get_corpus, get_corpus_embeddings, query, retrieval_augmented_query


logging.basicConfig(level=logging.INFO)

logging.info("Loading embedding model")
EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_REPO_NAME)
EMBEDDING_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_REPO_NAME)
EMBEDDING_MODEL.eval()
EMBEDDING_MODEL.cuda(EMBEDDING_MODEL_DEVICE)

logging.info("Loading corpus")
df_chunk_embs = get_corpus(DATA_FOLDER)

logging.info("Loading embeddings")    
chunk_embs = get_corpus_embeddings(df_chunk_embs)


logging.info("Loading meta data")
df_data = pd.read_parquet(DATA_FOLDER / PROCESSED_DATA_FILE)

@cl.on_chat_start
async def start():    
    # Store the information in the user session
    cl.user_session.set("df_chunk_embs", df_chunk_embs)
    cl.user_session.set("chunk_embs", chunk_embs)    
    cl.user_session.set("df_data", df_data)

    # settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="ChatMode",
                label="Chat Mode",
                values=CHAT_MODES,
                initial_index=0,
            ),
            Select(
                id="LLM",
                label="LLM",
                values=LLMS,
                initial_index=0,
            )
        ]
    ).send()
    cl.user_session.set("chat_mode", settings["ChatMode"])
    cl.user_session.set("llm_model_name", settings["LLM"])    
    logging.info("Chat session started")

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    cl.user_session.set("chat_mode", settings["ChatMode"])
    cl.user_session.set("llm_model_name", settings["LLM"])

@cl.on_message
async def main(user_msg):                
    llm_model_name = cl.user_session.get("llm_model_name")
    if cl.user_session.get("chat_mode") == "Plain":
        model_answer, references = await cl.make_async(query)(user_msg.content, llm_model_name), ""    
    elif cl.user_session.get("chat_mode") == "RAG":        
        df_chunk_embs = cl.user_session.get("df_chunk_embs")
        chunk_embs = cl.user_session.get("chunk_embs")
        df_data = cl.user_session.get("df_data")
        model_answer, references = await cl.make_async(retrieval_augmented_query)(user_msg.content, llm_model_name, df_data, df_chunk_embs, chunk_embs, EMBEDDING_MODEL, EMBEDDING_TOKENIZER)

    if isinstance(references, list):
        references = "* " + "\n* ".join(references) + "\n"
    
    res_text = f"{model_answer} \n\n### References:\n {references}"

    await cl.Message(content=res_text).send()
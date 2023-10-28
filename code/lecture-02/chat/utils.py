import logging

import numpy as np
import pandas as pd
import requests
import torch

from chat.constants import CHUNKED_EMB_FILE, WORKER_ADDR, QUERY_PROMPT, RETRIEVAL_TOP_K, CONTEXT_PROMPT_TEMPLATE

@torch.inference_mode()
def get_embedding(model, tokenizer, text, pooling="mean"):
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).cpu().numpy()
    
    batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(model.device)
    outputs = model(**batch_dict)

    if pooling == "mean":
        return average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    elif pooling == "last":
        return outputs.last_hidden_state[:, -1, :]
    else:
        raise NotImplementedError(f"{pooling} pooling not implemented")

def get_corpus(output_folder):
    # filter
    output_file_path = output_folder / CHUNKED_EMB_FILE
    df_chunk_embs = pd.read_parquet(output_file_path)
    df_chunk_embs = df_chunk_embs.drop_duplicates(subset=["text_chunked"]).reset_index(drop=True)
    return df_chunk_embs

def get_corpus_embeddings(df_chunk_embs):
    chunk_embs = np.vstack(df_chunk_embs.emb.values)
    # = df_chunk_embs.emb.values / np.linalg.norm(, axis=1, keepdims=True)
    chunk_embs = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    return chunk_embs

def dense_retrieval(query, chunk_embs, embedding_model, embedding_tokenizer, top_k):
    emb = get_embedding(embedding_model, embedding_tokenizer, QUERY_PROMPT.format(query=query), pooling="mean").ravel()
    query_emb = emb / np.linalg.norm(emb)
    res_ids = np.argsort(query_emb @ chunk_embs.T)[-top_k:][::-1]
    return res_ids

def get_urls(keys, df_data):
    urls = []
    for key in keys:
        urls.append(df_data[df_data.key == key].url.values[0])
    return urls

def query(user_msg, llm_model_name):
    payload = {"user_msg": user_msg}
    logging.info(f"Querying model {llm_model_name}")
    if llm_model_name == "vicuna-13b-v1.5":
        print(payload)
        res = requests.post(WORKER_ADDR + "/generate_stream", headers=[], json=payload).content.decode("utf-8")
    else:
        raise ValueError(f"Model {llm_model_name} not supported.")
    return res

def get_documents(dense_retrieval_results, df_chunk_embs):
    candidate_doc_ids = dense_retrieval_results
    candidate_docs = df_chunk_embs.iloc[candidate_doc_ids].text_chunked.values
    candidate_keys = df_chunk_embs.iloc[candidate_doc_ids].key.values
    return candidate_keys, candidate_docs


def context_retrieve(user_msg, df_data, df_chunk_embs, chunk_embs, emb_model, emb_tokenizer, top_k=RETRIEVAL_TOP_K):
    logging.info("Retrieving documents...")    
    logging.info("dense retrieval")
    dense_retrieval_results = dense_retrieval(user_msg, chunk_embs, emb_model, emb_tokenizer, top_k)
    candidate_keys, candidate_docs = get_documents(dense_retrieval_results, df_chunk_embs)
    candidate_urls = get_urls(candidate_keys, df_data)

    return candidate_docs, candidate_urls

def retrieval_augmented_query(user_msg, llm_model_name, df_data, df_chunk_embs, chunk_embs, emb_model, emb_tokenizer):
    # Retrieve context
    contexts, references = context_retrieve(user_msg, df_data, df_chunk_embs, chunk_embs, emb_model, emb_tokenizer)
    context_augmented_user_msg = CONTEXT_PROMPT_TEMPLATE.format(context_str="\n\n".join(contexts), query_str=user_msg)
    print(context_augmented_user_msg)
    model_answer = query(context_augmented_user_msg, llm_model_name)
    return model_answer, references
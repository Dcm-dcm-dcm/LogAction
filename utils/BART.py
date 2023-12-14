import numpy as np
from tqdm import tqdm
from transformers import BartTokenizer, BartModel
import torch



print("Is CUDA available? ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_type = 'CLS' #mean/CLS
encoder_name = 'bart-large' # bart-base/bart-large
HF_cache_dir = "./cached_transformers/"
tok = BartTokenizer.from_pretrained("facebook/{}".format(encoder_name), cache_dir=HF_cache_dir)
model = BartModel.from_pretrained("facebook/{}".format(encoder_name), cache_dir=HF_cache_dir)
batch_size = 1024

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def decode(tok, model, corpus):
    embeddings = []

    if tok:
        print("Using non Sentence Transformer models")
        for corpus_tmp in tqdm(chunks(corpus, batch_size)):
            encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
            sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
            sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(device), torch.LongTensor(attn_mask).to(
                device)

            with torch.no_grad():
                embedding_output_batch = model(sentence_batch, attn_mask)
                if embed_type == 'mean':
                    sentence_embeddings = mean_pooling(embedding_output_batch, attn_mask)
                elif embed_type == 'CLS':
                    sentence_embeddings = embedding_output_batch[0][:, 0, :]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())

            del sentence_batch, attn_mask, embedding_output_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Using Sentence Transformer models")
        for corpus_tmp in tqdm(chunks(corpus, batch_size)):
            sentence_embeddings = model.encode(corpus_tmp)
            embeddings.append(sentence_embeddings)

    return np.concatenate(embeddings, axis=0)

def bart_encode(corpus):
    model.to(device)
    X = decode(tok, model, corpus)
    return X

if __name__ == "__main__":
    corpus = ['hello you,12','nice to','yes man']
    X = bart_encode(corpus)
    a = 0



import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler



# LOAD DATASET
df = pd.read_csv('dataset.csv')


sequences = df['Sequence'].tolist()


for i in range(len(sequences)):
    if len(sequences[i]) > 1024:
        sequences[i] = sequences[i][:1024]

labels = df[['Chromatin', 'Cytoplasm', 'Cytosol', 'Exosome', 'Membrane', 'Nucleolus','Nucleoplasm' , 'Nucleus', 'Ribosome' ]].values


from multimolecule import RnaTokenizer, RiNALMoModel
import torch

DEVICE = 'cuda'
device = 'cuda'

tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model_llm = RiNALMoModel.from_pretrained('multimolecule/rinalmo')
model_llm.to(DEVICE)


def convert_seqs_to_embeddings(text):
    input = tokenizer(text, return_tensors='pt', padding=True)
    input = input.to(DEVICE)

    with torch.no_grad():
        output = model_llm(**input)
    emb = output.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (seq_length, embed_dim)
    token_embs = emb[1:-1, :]  # remove special tokens, if applicable

    # Aggregate using mean pooling
    sequence_embedding = token_embs.mean(axis=0)
    return sequence_embedding



def getSequenceEmbeddingsBatch(sequences, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i+batch_size]
        # Tokenize the batch with padding
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        inputs = inputs.to(DEVICE)
        
        with torch.no_grad():
            output = model_llm(**inputs)
        # Get the token embeddings (shape: batch_size x seq_length x embed_dim)
        emb_batch = output.last_hidden_state.cpu().numpy()
        
        # For each sequence in the batch, pool the token embeddings (excluding special tokens)
        for j in range(emb_batch.shape[0]):
            token_embs = emb_batch[j, 1:-1, :]  # remove special tokens if applicable
            seq_emb = token_embs.mean(axis=0)    # mean pooling for a single embedding per sequence
            all_embeddings.append(seq_emb)
        
        # Clear cache to manage GPU memory
        torch.cuda.empty_cache()
        
    return np.stack(all_embeddings)


embeddings = getSequenceEmbeddingsBatch(sequences, batch_size=32)
np.save("emb_data.npy", embeddings)
print("Embeddings saved successfully!")
exit()


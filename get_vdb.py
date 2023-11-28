import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def constructVDB(file_paths):
    chunks = []
    for filename in file_paths:
        with open(filename, 'r') as f:
            content = f.read()
            for chunk in chunkstring(content, 730):
                chunks.append(chunk)
    chunk_df = pd.DataFrame(chunks, columns=['chunk'])
    # Convert text chunks to embeddings using the pretrained SentenceTransformer model
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    embeddings = model.encode(chunk_df['chunk'].tolist())

    # Convert embeddings to a dataframe
    embedding_df = pd.DataFrame(embeddings.tolist())

    # Concatenate the original dataframe with the embeddings
    paraphrase_embeddings_df = pd.concat([chunk_df, embedding_df], axis=1)

    # Create a vector database from embeddings
    embeddings = paraphrase_embeddings_df.iloc[:, 1:].values  # All columns except the first (chunk text)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)  # Ensure that the array is C-contiguous

    # Preparation for Faiss
    dimension = embeddings.shape[1]  # the dimension of the vector space
    index = faiss.IndexFlatL2(dimension)
    # Normalize the vectors
    faiss.normalize_L2(embeddings)
    # Build the index
    index.add(embeddings)

    return paraphrase_embeddings_df, index


#def func2_updateVDB():
    
def main():
    files = glob.glob('knowledgeMaterials/**/*.md', recursive=True)
    df, idx = constructVDB(files)
    df.to_csv('paraphrase_embeddings_df.csv')
    faiss.write_index(idx, "faiss_index.idx")

main()


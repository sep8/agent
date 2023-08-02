import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
import os
import logging
import sys
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set environment variables
os.environ["HF_KEY"] = "your huggingface token"

# Initialize LLM
llm = ChatGroq(groq_api_key="please add your grok token", model_name="gemma2-9b-it")

# Load and process documents
loader = PyPDFLoader(r".\data\Algorithmic-Trading.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(doc_splits, embeddings)

def cosine_similarity_score(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]

def rag_with_weighted_ranking(query, k=5, faiss_weight=0.5, cosine_weight=0.5):
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search
    results = vectordb.similarity_search_with_score(query, k=k)
    
    matched_chunks = []
    faiss_distances = []
    cosine_similarities = []
    
    for doc, faiss_distance in results:
        # Embed the chunk
        chunk_embedding = embeddings.embed_query(doc.page_content)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity_score(np.array(query_embedding), np.array(chunk_embedding))
        
        matched_chunks.append({
            'chunk': doc.page_content,
            'faiss_distance': faiss_distance,
            'cosine_similarity': cosine_sim,
            'metadata': doc.metadata
        })
        faiss_distances.append(faiss_distance)
        cosine_similarities.append(cosine_sim)
    
    # Normalize scores
    normalized_faiss = normalize_scores(faiss_distances)
    normalized_cosine = normalize_scores(cosine_similarities)
    
    # Calculate weighted scores
    for i, chunk in enumerate(matched_chunks):
        weighted_score = (faiss_weight * (1 - normalized_faiss[i])) + (cosine_weight * normalized_cosine[i])
        chunk['weighted_score'] = weighted_score
    
    # Sort by weighted score
    ranked_chunks = sorted(matched_chunks, key=lambda x: x['weighted_score'], reverse=True)[:3]
    
    return ranked_chunks

# Function to combine top chunks for RAG
def get_combined_context(results):
    return " ".join([result['chunk'] for result in results])

prompt = hub.pull("rlm/rag-prompt")

# Chain
rag_chain = prompt | llm | StrOutputParser()

done=""
#Run
while not done:
    query= str(input("Please Enter the User Query "))
    query = "How can we use Social media and twitter for trading?"
    results = rag_with_weighted_ranking(query, k=5, faiss_weight=0.7, cosine_weight=0.3)
    combined_context = get_combined_context(results)
    generation = rag_chain.invoke({"context": combined_context, "question": query})
    print(generation)
    done = input("End the chat? (y/n): ") == "y"
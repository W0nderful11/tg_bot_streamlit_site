import streamlit as st
import chromadb
import os
import hashlib
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from chromadb.errors import UniqueConstraintError
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
import asyncio
import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
from collections import Counter
from wordcloud import WordCloud
import pandas as pd

# ---------------------------
# ChromaDB and Embedding Setup
# ---------------------------
DB_DIRECTORY = os.path.join(os.getcwd(), "search_db")
os.makedirs(DB_DIRECTORY, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIRECTORY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


class ChromaDBEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embedding_model.encode(input).tolist()


embedding = ChromaDBEmbeddingFunction(embedding_model)

try:
    collection = chroma_client.create_collection(
        name="rag_collection",
        metadata={"description": "RAG Collection with Ollama"},
        embedding_function=embedding
    )
except UniqueConstraintError:
    collection = chroma_client.get_collection("rag_collection")

MODEL_NAME = "llama3.2"


# ---------------------------
# Helper Functions
# ---------------------------
def compute_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def add_documents(documents, ids):
    collection.add(documents=documents, ids=ids)


def query_documents(query_text, n_results=None):
    if n_results is None:
        results = collection.get()
        return results["documents"]
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"]


def ollama_generate(prompt):
    llm = OllamaLLM(model=MODEL_NAME)
    return llm.invoke(prompt)


async def scrape_page(url):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


async def scrape_pages(urls):
    tasks = [scrape_page(url) for url in urls]
    return await asyncio.gather(*tasks)


def extract_keywords(query, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([query])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]


@st.cache_data
def search_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        urls = [result['href'] for result in results if 'href' in result]
        contents = asyncio.run(scrape_pages(urls))

        content_list = []
        metadata_list = []

        for i, content in enumerate(contents):
            metadata_list.append({
                "source": urls[i],
                "content": content
            })
            content_list.append(content)
            # Optionally, add the web content to your collection
            collection.add(
                documents=[content],
                metadatas=[{"url": urls[i]}],
                ids=[f"{hash(content)}"]
            )

        return content_list, metadata_list
    except Exception as e:
        return [], []


def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        text = None
    return text


def upload_file():
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            text_content = process_uploaded_file(uploaded_file)
            if text_content:
                doc_hash = compute_hash(text_content)
                existing_docs = collection.get()
                existing_hashes = [compute_hash(doc) for doc in existing_docs['documents']]

                if doc_hash in existing_hashes:
                    st.warning("This document is already in the database.")
                else:
                    doc_id = f"doc{len(existing_docs['documents']) + 1}"
                    embeddings = embedding([text_content])[0]
                    collection.add(
                        ids=[doc_id],
                        documents=[text_content],
                        embeddings=[embeddings]
                    )
                    st.success(f"File processed and saved with ID: {doc_id}")
            else:
                st.error("Unsupported file type or empty file.")
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ---------------------------
# RAG Pipeline Functions
# ---------------------------
def query_chromadb(query_text, n_results=1):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"], results.get("metadatas", [])


def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(query_text)
    if retrieved_docs and retrieved_docs[0]:
        context = " ".join(retrieved_docs[0])
    else:
        context = "No relevant documents found."
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = ollama_generate(augmented_prompt)
    return response


def query_processor(user_query, use_web_search=False):
    try:
        if use_web_search:
            st.info("Performing web search...")
            search_results, metadata = search_duckduckgo(user_query)
            context = "\n".join(search_results) if search_results else "No web results found."
            response = ollama_generate(f"Context: {context}\n\nQuestion: {user_query}\nAnswer:")
            return response, metadata
        else:
            response = rag_pipeline(user_query)
            return response, []
    except Exception as e:
        return f"Error: {str(e)}", []


# ---------------------------
# Word Cloud Visualization
# ---------------------------
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("AI-Powered Chatbot")

    # Sidebar navigation
    menu = ["Home", "View Documents", "Add Document", "Ask Ollama", "Visualizations"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("Welcome to the AI-Powered Knowledge Hub")
        st.write(
            """
            Welcome to **AI OllamaLLM-based chatbot**, a collaborative platform designed to help you 
            store, explore, and interact with knowledge efficiently.

            **What You Can Do Here**  
            - 📂 **Upload Documents** – Add PDFs, text files, or web links to the knowledge base.  
            - 🤖 **Ask AI-Powered Questions** – Get instant insights using advanced AI models.  
            - 📊 **Visualize Data** – Discover trends with interactive graphs, word clouds, and topic models.  
            - 🛠 **Collaborate & Improve** – Suggest edits, contribute knowledge, and enhance the database together.  

            **Get Started**  
            Simply **upload a document** or **ask a question**, and let AI assist you.  

            Ready to explore? **Start now!** 🚀  
            """
        )
        st.text("Navigate using the sidebar to interact with the app.")

    elif choice == "View Documents":
        if st.button("Delete all documents"):
            try:
                collection.delete(where={"id": {"$ne": ""}})
                st.sidebar.success("All documents deleted successfully.")
            except Exception as e:
                st.sidebar.error(f"Error deleting documents: {e}")
        docs = collection.get()
        if docs['documents']:
            for i, (doc, doc_id) in enumerate(zip(docs['documents'], docs['ids'])):
                with st.expander(f"Document {i + 1} (ID: {doc_id})"):
                    st.write(doc)
                    if st.button(f"Hide Document {i + 1}", key=f"hide_{doc_id}"):
                        collection.delete(ids=[doc_id])
                        st.success(f"Document {i + 1} (ID: {doc_id}) deleted.")
            # Button to trigger PCA visualization
            if st.button("Show PCA Visualisation"):
                # Combine all stored documents' text into one string
                all_text = " ".join(docs['documents'])
                # Extract words using regex and convert to lowercase
                words = re.findall(r'\w+', all_text.lower())
                # Get the top 100 most common words
                word_counts = Counter(words)
                top_words = [word for word, count in word_counts.most_common(100)]
                if not top_words:
                    st.warning("No words found for visualization.")
                else:
                    # Compute embeddings for the selected words
                    word_embeddings = embedding(top_words)
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(word_embeddings)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
                    for i, word in enumerate(top_words):
                        ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9)
                    st.pyplot(fig)
        else:
            st.warning("No documents found.")

    elif choice == "Add Document":
        st.subheader("Add Document")
        input_method = st.radio("Select input method", ["Upload File", "Enter Text"], index=0)
        if input_method == "Upload File":
            upload_file()
        else:
            text_input = st.text_area(
                "Enter your document content here:",
                placeholder="Type your document content..."
            )
            if st.button("Submit Text"):
                if text_input:
                    doc_hash = compute_hash(text_input)
                    existing_docs = collection.get()
                    existing_hashes = [compute_hash(doc) for doc in existing_docs["documents"]]
                    if doc_hash in existing_hashes:
                        st.warning("This text is already in the database.")
                    else:
                        doc_id = f"doc{len(existing_docs['documents']) + 1}"
                        embeddings = embedding([text_input])[0]
                        collection.add(
                            ids=[doc_id],
                            documents=[text_input],
                            embeddings=[embeddings]
                        )
                        st.success(f"Text processed and saved with ID: {doc_id}")
                else:
                    st.error("Please enter some text before submitting.")

    elif choice == "Ask Ollama":
        user_input = st.text_input("Enter your query:")
        use_web = st.checkbox("Search the web", value=False)
        if user_input:
            with st.spinner("Processing your query..."):
                response, metadata = query_processor(user_input, use_web_search=use_web)
                st.write("**Response:**")
                st.write(response)
                if metadata:
                    st.write("**Sources:**")
                    for meta in metadata:
                        with st.expander(f"Source: {meta.get('source', 'Unknown source')}"):
                            st.write(meta.get("content", "No content extracted"))

    elif choice == "Visualizations":
        st.subheader("Data Visualizations")
        docs = collection.get()
        if docs['documents']:
            all_text = " ".join(docs['documents'])
            st.write("### Word Cloud")
            generate_word_cloud(all_text)
        else:
            st.warning("No documents found for visualization.")


if __name__ == "__main__":
    main()

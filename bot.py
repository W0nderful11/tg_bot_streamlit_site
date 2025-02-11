from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext
import os
import hashlib
import asyncio
import httpx
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
import numpy as np
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import UniqueConstraintError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from collections import Counter

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

@staticmethod
def extract_keywords(query, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([query])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

async def search_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        urls = [result['href'] for result in results if 'href' in result]
        # Используем await для асинхронного выполнения
        contents = await scrape_pages(urls)
        content_list = []
        metadata_list = []
        for i, content in enumerate(contents):
            metadata_list.append({
                "source": urls[i],
                "content": content
            })
            content_list.append(content)
            # Опционально добавляем данные в коллекцию
            collection.add(
                documents=[content],
                metadatas=[{"url": urls[i]}],
                ids=[f"{hash(content)}"]
            )
        return content_list, metadata_list
    except Exception as e:
        return [], []

def process_uploaded_file(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = None
    return text

def upload_file_from_path(file_path):
    text_content = process_uploaded_file(file_path)
    if text_content:
        doc_hash = compute_hash(text_content)
        existing_docs = collection.get()
        existing_hashes = [compute_hash(doc) for doc in existing_docs["documents"]]
        if doc_hash in existing_hashes:
            return "This document is already in the database."
        else:
            doc_id = f"doc{len(existing_docs['documents']) + 1}"
            embeddings = embedding([text_content])[0]
            collection.add(
                ids=[doc_id],
                documents=[text_content],
                embeddings=[embeddings]
            )
            return f"File processed and saved with ID: {doc_id}"
    else:
        return "Unsupported file type or empty file."

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

async def query_processor(user_query, use_web_search=False):
    try:
        if use_web_search:
            # Выполняем веб-поиск
            search_results, metadata = await search_duckduckgo(user_query)
            context = "\n".join(search_results) if search_results else "No web results found."
            response = ollama_generate(f"Context: {context}\n\nQuestion: {user_query}\nAnswer:")
            return response, metadata
        else:
            # Используем локальный RAG-пайплайн
            response = rag_pipeline(user_query)
            return response, []
    except Exception as e:
        return f"Error: {str(e)}", []


# ---------------------------
# Telegram Bot Handlers
# ---------------------------
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Welcome to the AI Chatbot!\n"
        "Commands:\n"
        "/chat <your question> - Ask Ollama (local RAG)\n"
        "/web <your question> - Search the web and answer\n"
        "/upload <file path> - Upload a file (PDF or TXT)\n"
        "/pca - Generate PCA visualization of stored document word embeddings"
    )


async def chat_command(update: Update, context: CallbackContext):
    user_query = " ".join(context.args)
    if user_query:
        response, _ = query_processor(user_query, use_web_search=False)
        await update.message.reply_text(response)
    else:
        await update.message.reply_text("Please provide a question after /chat.")


async def web_command(update, context):
    try:
        user_query = " ".join(context.args)
        if not user_query:
            await update.message.reply_text("Please provide a search query.")
            return

        # Добавляем await, чтобы дождаться выполнения query_processor
        response, metadata = await query_processor(user_query, use_web_search=True)

        # Форматируем ответ для пользователя
        result_message = f"Response:\n{response}\n\nSources:\n"
        for meta in metadata:
            result_message += f"- {meta['source']}\n"

        await update.message.reply_text(result_message)
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")



async def upload_command(update: Update, context: CallbackContext):
    file_path = " ".join(context.args)
    if file_path:
        result = upload_file_from_path(file_path)
        await update.message.reply_text(result)
    else:
        await update.message.reply_text("Please provide the file path after /upload.")


async def pca_command(update: Update, context: CallbackContext):
    docs = collection.get()
    if not docs["documents"]:
        await update.message.reply_text("No documents found to generate PCA visualization.")
        return

    all_text = " ".join(docs["documents"])
    words = re.findall(r'\w+', all_text.lower())
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(100)]

    if not top_words:
        await update.message.reply_text("No words found for PCA visualization.")
        return

    word_embeddings = embedding(top_words)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(word_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    for i, word in enumerate(top_words):
        ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    ax.set_title("PCA Visualization of Word Embeddings")

    plot_filename = "pca_plot.png"
    fig.savefig(plot_filename)
    plt.close(fig)

    with open(plot_filename, "rb") as photo:
        await update.message.reply_photo(photo=photo, caption="PCA Visualization of Word Embeddings")
    os.remove(plot_filename)


    # Combine all stored documents' text
    all_text = " ".join(docs["documents"])
    # Extract words using regex (convert to lowercase)
    words = re.findall(r'\w+', all_text.lower())
    # Get the top 100 most common words
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(100)]
    if not top_words:
        update.message.reply_text("No words found for PCA visualization.")
        return

    # Compute embeddings for the selected words
    word_embeddings = embedding(top_words)
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(word_embeddings)

    # Create scatter plot with annotations
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    for i, word in enumerate(top_words):
        ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    ax.set_title("PCA Visualization of Word Embeddings")

    # Save the plot to a temporary file
    plot_filename = "pca_plot.png"
    fig.savefig(plot_filename)
    plt.close(fig)

    # Send the image back to the user
    with open(plot_filename, "rb") as photo:
        update.message.reply_photo(photo=photo, caption="PCA Visualization of Word Embeddings")
    os.remove(plot_filename)

def main():
    application = Application.builder().token("7457156728:AAE-x8buJY1I84ieH24HjFJxkh2j7T0ZECA").build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("chat", chat_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("upload", upload_command))
    application.add_handler(CommandHandler("pca", pca_command))

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()

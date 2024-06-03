import openai
import replicate
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import io
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

openai.api_key = "KEY"
replicate_api_token = 'KEY'

replicate_client = replicate.Client(api_token=replicate_api_token)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384  
indexing = faiss.IndexFlatL2(dimension)
texts = []  

texts_file = 'texts_cache.pkl'
embeddings_file = 'embeddings_cache.pkl'

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def get_embedding(text):
    return embedding_model.encode(text)

def scrape_website(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        texts = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table', 'tr', 'th', 'td', 'span', 'div', 'a', 'b', 'strong', 'i', 'em'])]

        sub_page_links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if 'http' not in a['href'] or url in a['href']]
        for link in sub_page_links:
            sub_page_texts = scrape_sub_page(link)
            texts.extend(sub_page_texts)
            
        return texts
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

def scrape_sub_page(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        sub_page_texts = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table', 'tr', 'th', 'td', 'blockquote', 'span', 'div', 'a', 'b', 'strong', 'i', 'em'])]
        
        pdf_links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
        for pdf_link in pdf_links:
            pdf_text = scrape_pdf(pdf_link)
            sub_page_texts.append(pdf_text)
            
        return sub_page_texts
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

def scrape_pdf(pdf_url):
    try:
        response = session.get(pdf_url, timeout=10)
        response.raise_for_status()
        with io.BytesIO(response.content) as open_pdf_file:
            reader = PdfReader(open_pdf_file)
            pdf_text = ""
            for page_num in range(len(reader.pages)):
                pdf_text += reader.pages[page_num].extract_text()
            return pdf_text
    except (requests.exceptions.RequestException, PdfReadError) as e:
        print(f"Error scraping PDF {pdf_url}: {e}")
        return ""


def setup_vector_db():
    global texts, indexing
    # we check if we already cached the data
    if os.path.exists(texts_file) and os.path.exists(embeddings_file):
        with open(texts_file, 'rb') as tf, open(embeddings_file, 'rb') as ef:
            texts = pickle.load(tf)
            embeddings = pickle.load(ef)
        print("Loaded texts and embeddings from cache.")
    else:
        # scraping the data and creating embeddings
        urls = [
            "https://u.ae/en/information-and-services",
            "https://u.ae/en/information-and-services/visa-and-emirates-id",
            "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas",
            "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa"
        ]
        for url in urls:
            texts.extend(scrape_website(url))
        
        if not texts:
            print("No texts scraped from the websites.")
            return

        print(f"Scraped {len(texts)} texts.")
        embeddings = np.array([get_embedding(text) for text in texts])

        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            print(f"Embeddings shape is not correct: {embeddings.shape}")
            return
        
        # i saved the texts and embeddings to cache to save time 
        with open(texts_file, 'wb') as tf, open(embeddings_file, 'wb') as ef:
            pickle.dump(texts, tf)
            pickle.dump(embeddings, ef)
        print("Saved texts and embeddings to cache.")

    indexing.add(embeddings)
    print(f"Vector database setup complete with {indexing.ntotal} embeddings.")

setup_vector_db()

def get_relevant_texts(query):
    global indexing, texts
    query_embedding = get_embedding(query).astype('float32')
    print(f"Query embedding shape: {query_embedding.shape}")
    D, I = indexing.search(np.array([query_embedding]), 5)
    print(f"FAISS search results - Distances: {D}, Indices: {I}")

    relevant_indices = I[0]
    print(f"Relevant indices: {relevant_indices}")
    if I.size == 0 or any(i >= len(texts) for i in relevant_indices):
        print("FAISS search returned out of range index.")
        return ["No relevant texts found."]
    
    unique_texts = list(set([texts[i] for i in relevant_indices if i < len(texts)]))
    print(f"Unique relevant texts: {unique_texts}")
    return unique_texts

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('new_query')
def handle_new_query(data):
    user_query = data['query']
    print(f"Received new query: {user_query}")
    relevant_texts = get_relevant_texts(user_query)
    new_prompt = f"Search results: {relevant_texts}\nUser query: {user_query} (Do not answer if query does not relate to the search results)"
    
    socketio.start_background_task(target=query_llms, prompt=new_prompt, relevant_texts=relevant_texts)

def query_llms(prompt, relevant_texts):
    print(f"Querying LLMs with prompt: {prompt}")
    gpt35_response = query_gpt35(prompt)
    socketio.emit('response', {'model': 'gpt35', 'response': gpt35_response})
    
    gpt4_response = query_gpt4(prompt)
    socketio.emit('response', {'model': 'gpt4', 'response': gpt4_response})
    
    llama2_response = query_llama2(prompt)
    socketio.emit('response', {'model': 'llama2', 'response': llama2_response})
    
    falcon_response = query_falcon(prompt)
    socketio.emit('response', {'model': 'falcon', 'response': falcon_response})
    
    best_response = evaluate_responses({
        'gpt35': gpt35_response,
        'gpt4': gpt4_response,
        'llama2': llama2_response,
        'falcon': falcon_response
    }, relevant_texts)
    socketio.emit('best_response', {'response': best_response})


def query_gpt35(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answers to questions that relate to the search results given if it doesnt pertain it apologize and dont answer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

def query_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answers to questions that relate to the search results given if it doesnt pertain it apologize and dont answer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

def clean_response(response):
    cleaned_response = response.replace(", ", " ").replace(",", " ")
    cleaned_response = " ".join(cleaned_response.split())
    return cleaned_response

def query_llama2(prompt):
    output = replicate_client.run(
        "meta/llama-2-7b-chat",
        input={"prompt": prompt}
    )
    result = clean_response("".join(output))
    return result

def query_falcon(prompt):
    output = replicate_client.run(
        "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
        input={"prompt": prompt}
    )
    result = clean_response("".join(output))
    return result

def evaluate_responses(responses, relevant_texts):
    # we compute embeddings for relevant texts
    relevant_embeddings = np.array([get_embedding(text) for text in relevant_texts])
    
    # we compute the average cosine similarity of each model to the relevant texts
    similarities = {}
    for model, response in responses.items():
        response_embedding = get_embedding(response)
        cos_sim = cosine_similarity([response_embedding], relevant_embeddings).mean()
        similarities[model] = cos_sim

    # we then find the model with the highest average cosine similarity
    best_model = max(similarities, key=similarities.get)
    return f"Best response by {best_model}: {responses[best_model]}"

if __name__ == '__main__':
    socketio.run(app, debug=True)
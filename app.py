import openai
import replicate
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from sentence_transformers import SentenceTransformer
import torch
import time
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

openai.api_key = "API-KEY"
replicate_api_token = 'API-KEY'

replicate_client = replicate.Client(api_token=replicate_api_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

dimension = 384
indexing = faiss.IndexFlatL2(dimension)
texts = []

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

def get_embeddings(texts):
    batch_size = 64 
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_inputs = embedding_model.encode(batch_texts, convert_to_tensor=True, device=device)
            embeddings.append(encoded_inputs.cpu().numpy())
    
    return np.vstack(embeddings)

def save_embeddings(texts, embeddings, file_path='embeddings_cache.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump((texts, embeddings), f)

def load_embeddings(file_path='embeddings_cache.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return [], None

def scrape_site(url, visited_urls, start_time, timeout, urls):
    if url in visited_urls:
        return []
    visited_urls.add(url)
    
    print(f"Scraping: {url}")
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []
    
    page_texts = []
    tags_to_scrape = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table', 'tr', 'th', 'td', 'span', 'div', 'a', 'b', 'strong', 'i', 'em']
    for tag in tags_to_scrape:
        elements = soup.find_all(tag)
        for element in elements:
            text = element.get_text().strip()
            if text:
                page_texts.append(text)
    
    links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True)]
    for link in links:
        if link.startswith("https://u.ae/en/information-and-services"):
            if (time.time() - start_time) > timeout:
                print("Scraping timed out. Moving to next URL.")
                return page_texts
            page_texts.extend(scrape_site(link, visited_urls, start_time, timeout, urls))
    
    return page_texts

def setup_vector_db():
    global texts, indexing
    cache_file = 'embeddings_cache.pkl'
    texts, embeddings = load_embeddings(cache_file)
    
    if texts is None or embeddings is None:
        print("No cache found. Starting scraping and embedding process.")
        urls = [
            "https://u.ae/en/information-and-services",
        ]
        visited_urls = set()
        start_time = time.time()
        timeout = 1200

        for url in urls:
            texts.extend(scrape_site(url, visited_urls, start_time, timeout, urls))
            if (time.time() - start_time) > timeout:
                print("Scraping process timed out. Moving to next URL.")
                start_time = time.time()  
        
        if not texts:
            print("No texts scraped from the websites.")
            return

        print(f"Scraped {len(texts)} texts.")
        embeddings = get_embeddings(texts)

        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            print(f"Embeddings shape is not correct: {embeddings.shape}")
            return
        
        save_embeddings(texts, embeddings, cache_file)
        print("Embeddings saved to cache.")
    else:
        print("Loaded embeddings from cache.")
    
    indexing.add(embeddings)
    print(f"Vector database setup complete with {indexing.ntotal} embeddings.")

setup_vector_db()

def get_relevant_texts(query):
    global indexing, texts
    query_embedding = get_embeddings([query]).astype('float32')
    print(f"Query embedding shape: {query_embedding.shape}")
    D, I = indexing.search(query_embedding, 5)
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
            {"role": "system", "content": "You are a helpful assistant that only answers to questions that relate to the search results given. If it doesn't pertain, apologize and don't answer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

def query_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answers to questions that relate to the search results given. If it doesn't pertain, apologize and don't answer."},
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
    relevant_embeddings = get_embeddings(relevant_texts)
    similarities = {}
    for model, response in responses.items():
        response_embedding = get_embeddings([response])
        cos_sim = cosine_similarity(response_embedding, relevant_embeddings).mean()
        similarities[model] = cos_sim
    best_model = max(similarities, key=similarities.get)
    return f"Best response by {best_model}: {responses[best_model]}"

if __name__ == '__main__':
    socketio.run(app, debug=True)

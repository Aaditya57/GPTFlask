from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import PyPDF2
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Initialize Chroma index globally
chroma_index = None

# PDF parsing function
def pdf_parser(file_path):
    detected_text = ''
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        for page in pdf_reader.pages:
            detected_text += page.extract_text() + "\n"
    return detected_text

# Text splitting function
def splitter_character(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=2,
        length_function=len,
        separator=" ",
        is_separator_regex=True
    )
    texts = text_splitter.split_text(text)
    return texts


def splitter_recursive(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 0,
        length_function = len
    )
    texts = text_splitter.split_text(text)
    return texts


def combine_sentences(sentences, buffersize = 1):
    for i in range(len(sentences)):

        combined_sentence = ""

        for j in range (i-buffersize, i):
            if j>=0:
                combined_sentence += sentences[j]['sentence'] + " "
        combined_sentence +=sentences[i]['sentence']

        for j in range (i+1, i+1 + buffersize):
            if j<len(sentences):
                combine_sentence += ' ' + sentences[j]['sentences']

        sentences[i]['combined_sentence'] = combined_sentence 

    return sentences

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        sentences[i]['distance_to_next'] = distance

    return distances, sentences

def splitter_semantic(text):
    single_sentences_list = re.split(r'(?<=[.>?!])\s+', text)
    sentences = [{'sentence':x, 'index': y} for y, x in enumerate(single_sentences_list)]
    sentences = combine_sentences(sentences)
    embeddings = OpenAIEmbeddings.embed_documents([x['combined_sentence'] for x in sentences])

    for i,sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    distances, sentences = calculate_cosine_distances
        # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []
    breakpoint_percentile_threshold = 95
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list
    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        
        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    


    return combined_text



# Initialize Chroma vector store
def initialize_chroma(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="chroma_index")
    return vectorstore

# Similarity search function
def sim_search(message, k):
    global chroma_index
    if chroma_index is None:
        text = pdf_parser('TenStages.pdf')
        texts = splitter_semantic(text)
        chroma_index = initialize_chroma(texts)
    results = chroma_index.similarity_search(message, k)
    return [result.page_content for result in results]

# Flask app
def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/answer", methods=["POST"])
    def answer():
        data = request.get_json()
        message = str(data.get("message", ""))
        print("Received message:", message)
        
        contexts = sim_search(message, 5)
        print("Similar contexts:", contexts)

        full_context = "\n".join(contexts)


        prompt = PromptTemplate(template="{full_context}\n\nBased on the above, {message}", input_variables=["full_context", "message"])

        llm_chain = prompt | llm

        input = {
            'full_context':full_context,
            'message':message
        }

        response =  llm_chain.invoke(input=input).content

        return response
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import PyPDF2
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
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
        chunk_overlap=20,
        length_function=len,
        separator=" ",
        is_separator_regex=True
    )
    texts = text_splitter.split_text(text)
    return texts

def splitter_recursive(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len
    )
    texts = text_splitter.split_text(text)
    return texts

def splitter_semantic(text):
   text_splitter = SemanticChunker(OpenAIEmbeddings())
   texts = text_splitter.split_text(text)
   return texts  


# Initialize Chroma vector store
def initialize_chroma(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="chroma_index")
    return vectorstore

# Similarity search function
def sim_search_similarity(message, k, splittercase):
    global chroma_index
    if chroma_index is None:
        text = pdf_parser('TenStages.pdf')
        texts = splitter(text, splittercase)
        chroma_index = initialize_chroma(texts)
    results = chroma_index.similarity_search(message, k)
    return [result.page_content for result in results]

def sim_search_threshold(message, k, splittercase):
    global chroma_index
    if chroma_index is None:
        text = pdf_parser('Collection.pdf')
        texts = splitter(text, splittercase)
        chroma_index = initialize_chroma(texts)
    results = chroma_index.search(query = message, search_type = "similarity_score_threshold", score_threshold=0.9)
    return [result.page_content for result in results]

def sim_search_MMR(message, k, splittercase):
    global chroma_index
    if chroma_index is None:
        text = pdf_parser('Collection.pdf')
        texts = splitter(text, splittercase)
        chroma_index = initialize_chroma(texts)
    results = chroma_index.max_marginal_relevance_search(message, k, 25, 0.5)
    return [result.page_content for result in results]

def splitter(text, num):
    match num:
        case 1:
            return splitter_character(text)
        case 2:
            return splitter_recursive(text)
        case 3:
            return splitter_semantic(text)
        case default:
            return splitter_character(text)

def search(message, k, splittercase, searchcase):   
    match searchcase:
        case 1:
            return sim_search_similarity(message, k, splittercase)
        case 2:
            return sim_search_threshold(message, k, splittercase)
        case 3:
            return sim_search_MMR(message, k, splittercase)
        case default:
            return sim_search_similarity(message, k, splittercase)

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
        splitter = str(data.get("splitter", ""))
        retrieval = str(data.get("retrieval", ""))

        k = int(splitter)
        print (splitter)
        # message, number of outputs, splittercase, vectorretrivalcase
        # splittercase: char, recursive, semantic
        # retrivalcase: similarity, by threshold, and MMR
        # options are 1, 2, 3 (more intuitive that way)
        response = ""
        x = 3
        while (x != 4):
            contexts = search(message, 5, k, x)
            print(contexts)
            full_context = "\n".join(contexts)
            prompt = PromptTemplate(template="{full_context}\n\nBased on the above, {message}", input_variables=["full_context", "message"])
            llm_chain = prompt | llm
            input = {
                'full_context':full_context,
                'message':message
            }
            temp_response = llm_chain.invoke(input=input).content
            response =  response +"\n\n " + temp_response
            x+=1
        response =  response +"\n\n "   

        return response
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

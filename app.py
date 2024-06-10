from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import PyPDF2
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain

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
def splitter(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separator=" ",
        is_separator_regex=True
    )
    texts = text_splitter.split_text(text)
    return texts

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
        texts = splitter(text)
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
        message = data.get("message", "")
        print("Received message:", message)
        
        contexts = sim_search(message, 5)
        print("Similar contexts:", contexts)

        full_context = "\n".join(contexts)

        full_prompt = f"{full_context}\n\nBased on the above, {message}"

        llm_chain = load_qa_chain(llm, chain_type="map_reduce", input_key="context", output_key="answer")

        response = llm_chain.run({"context": full_context, "question": message})

        print(response)

        # full_response = response[0]["message"]["content"]
        
        # return jsonify({"response": full_response})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import PyPDF2

llm = OpenAI(model_name="gpt-3.5-turbo")
chroma_index = None

def pdf_parser():
    pdf_file_obj = open('TenStages.pdf', 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    detected_text = ''

    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num]
        detected_text += page_obj.extract_text() + "\n"

    pdf_file_obj.close()
    return detected_text


def splitter():
    text_splitter = CharacterTextSplitter(chunk_size=200, 
                                        chunk_overlap=50,
                                        length_function=len,
                                        separator= " ",
                                        is_separator_regex=True)
    texts = text_splitter.split_text(pdf_parser())
    return texts

def initialize_chroma(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="chroma_index")
    return vectorstore


texts = splitter()
#print(texts)
chroma_index = initialize_chroma(texts) 
message = "what is a genocide"
results = chroma_index.similarity_search(message, k=5)

contexts = [result.page_content for result in results]



def create_app():
    app = Flask(__name__)


    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/answer", methods=["POST"])
    def answer():
        texts = splitter()
        print(texts)
        chroma_index = initialize_chroma(texts)
        data = request.get_json()
        message = data["message"]

        # Search Chroma index
        results = chroma_index.similarity_search(message, k=7)
        print(results)
        contexts = [result.page_content for result in results]
        
        # Construct a query to OpenAI
        output = [
            {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": message}
        ]
        # for context in contexts:
        #     output.append({"role": "user", "content": context})

        response = llm.generate(output)  

        full_response = response['choices'][0]['message']['content']

        return full_response, {"Content-Type": "text/plain"}

    return app
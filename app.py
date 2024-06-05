from flask import Flask, render_template, request 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import PyPDF2

client = OpenAI()


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
                                        chunk_overlap=0,
                                        length_function=len, 
                                        separator=" ", 
                                        is_separator_regex=False)
    texts = text_splitter.split_text(pdf_parser())

texts = splitter()
print(len(texts))

embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/answer", methods=["POST"])
    def answer():
        data = request.get_json()
        message = data["message"]


        def generate():

            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user", "content":message}],
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield(chunk.choices[0].delta.content)

        return generate(), {"Content-Type": "text/plain"}
    
    return app
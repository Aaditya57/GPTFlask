from flask import Flask, render_template, request, jsonify
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader


client = OpenAI()    

def splitter():
    loader = PyPDFLoader('TenStages.pdf')
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, 
                                        chunk_overlap=0,
                                        length_function=len, 
                                        separator=" ")
    texts = text_splitter.split_documents(docs)
    return texts

texts = splitter()
print(len(texts))

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
document_search = vectorstore.as_retriever()

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/answer", methods=["POST"])
    def answer():
        data = request.get_json()
        message = data["message"]
        system_prompt = (
            ""
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )



        def generate():

            model = OpenAI(model = "gpt-3.5-turbo")

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{message}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(document_search, question_answer_chain)
            relevant_docs = document_search.get_relevant_documents(message)
            context = " ".join([doc.page_content for doc in relevant_docs])
            response = rag_chain({"context": context, "message": message})

            for chunk in response:
                yield chunk
        return generate(), {"Content-Type": "text/plain"}
    
    return app
from flask import Flask, render_template, request 
from openai import OpenAI

client = OpenAI()


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route ("upload_pdf", methods = ["POST"])
    def main():
        files = request.files["TenStages.pdf"]
        return convert_pdf_to_jpg(files,files.name)if __name__ == ‘__main__’:
        app.run(debug = True)

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
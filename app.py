
from flask import Flask, render_template, redirect, request

from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

load_dotenv()

client = OpenAI()

app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True
@app.route('/', methods = ["POST", "GET"])

def index():
    if request.method == "POST":
        
        prompt = request.form.get("prompt")
        if not prompt:
            return redirect("/")
        

        askAI(prompt)
        #print(output)
        return render_template("index.html")
    else:
        return render_template("index.html")
    
def askAI(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role":"user", "content": "limit yourself to one sentence: " + prompt}],
        stream=True,
    )
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
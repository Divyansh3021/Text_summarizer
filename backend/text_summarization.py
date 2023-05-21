from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def text_summarize(text):
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    text = text

    tokens = tokenizer(text, truncation = True, max_length = 500, return_tensors = "pt")

    summary = model.generate(**tokens)

    decoded_str = tokenizer.decode(summary[0])

    return decoded_str


from flask import Flask, request, redirect, session

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def welcome():
    return "Hello, welcome to summmarizer"

@app.route("/summarize", methods = ['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form.get('text')
        return text_summarize(text)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
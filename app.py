from flask import Flask, request, render_template_string
from flask_sqlalchemy import SQLAlchemy
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import openai

# 尝试加载模型，如果模型未安装则安装模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# 从环境变量中读取 OpenAI API 密钥
openai.api_key = os.getenv('OPENAI_API_KEY')

openai.api_key = 'YOUR_API_KEY'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///novel_assistant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Novel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    content = db.Column(db.Text)

class Character(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    description = db.Column(db.Text)
    novel_id = db.Column(db.Integer, db.ForeignKey('novel.id'))

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    summary = db.Column(db.Text)
    keywords = db.Column(db.Text)
    novel_id = db.Column(db.Integer, db.ForeignKey('novel.id'))

with app.app_context():
    db.create_all()

nlp = spacy.load('en_core_web_sm')

def analyze_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")

    summary = summarizer(parser.document, 5)
    summary = " ".join([str(sentence) for sentence in summary])

    key_words = nlp(text)
    key_words = [token.text for token in key_words if token.is_stop is False and token.is_punct is False]
    key_words = list(set(key_words))[:10]

    return summary, key_words

def generate_text(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Novel Assistant</title>
        </head>
        <body>
            <h1>Upload your novel</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
        </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        content = file.read().decode('utf-8')
        summary, key_words = analyze_text(content)

        new_novel = Novel(title=file.filename, content=content)
        db.session.add(new_novel)
        db.session.commit()

        new_analysis = Analysis(summary=summary, keywords=','.join(key_words), novel_id=new_novel.id)
        db.session.add(new_analysis)
        db.session.commit()

        return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Analysis Result</title>
            </head>
            <body>
                <h1>Analysis Result</h1>
                <h2>Summary</h2>
                <p>{{ summary }}</p>
                <h2>Keywords</h2>
                <ul>
                    {% for word in key_words %}
                    <li>{{ word }}</li>
                    {% endfor %}
                </ul>
            </body>
            </html>
        ''', summary=summary, key_words=key_words)

if __name__ == '__main__':
    app.run(debug=True)

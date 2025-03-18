import nltk
from flask import Flask, request, render_template
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import webbrowser
from threading import Timer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize Flask app
app = Flask(__name__)

# Function to open the browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# Text Preprocessing
def preprocess_text(text):
    """
    Tokenize and stem the input text.
    """
    sentences = sent_tokenize(text)  # Split text into sentences
    stemmer = PorterStemmer()  # Initialize stemmer
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]  # Tokenize each sentence
    stemmed_sentences = [[stemmer.stem(word) for word in sentence] for sentence in tokenized_sentences]  # Stem each word
    return stemmed_sentences, sentences  # Return both stemmed and original sentences

# Word Sense Disambiguation
def disambiguate_word(word, text):
    """
    Resolve the meaning of an ambiguous word using WordNet.
    """
    synset = lesk(text, word)
    if synset:
        return synset.definition()
    return "No specific meaning found."

def disambiguate_text(text):
    """
    Disambiguate all words in the text and return ambiguous words with their meanings.
    """
    ambiguous_words = {}
    for word in word_tokenize(text):
        if len(wn.synsets(word)) > 1:  # Check if the word is ambiguous
            meaning = disambiguate_word(word, text)
            ambiguous_words[word] = meaning  # Store word and its meaning
    return ambiguous_words

# Summarization using LexRank
def summarize_text(text, sentences_count=3):
    """
    Summarize the input text using Sumy's LexRank summarizer.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# Flask Web Interface
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']  # Get input text from the form
        _, original_sentences = preprocess_text(text)  # Preprocess the text
        ambiguous_words = disambiguate_text(text)  # Disambiguate words in the entire text
        summarized_text = summarize_text(text)  # Summarize the entire paragraph
        return render_template(
            'index.html',
            summary=summarized_text,
            ambiguous_words=ambiguous_words,  # Pass the dictionary
            input_sentences=original_sentences
        )
    return render_template('index.html')  # Render the form

# Run the Flask app
if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open the browser after 1 second
    app.run(debug=True)
import streamlit as st
from transformers import pipeline
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. Download NLTK data (run once) ---
# This downloads necessary data for keyword extraction.
# It might show a download window the very first time you run the app.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- 2. Load AI Models ---
# This loads a pre-trained summarization model. It will download a large file
# the very first time it runs, which can take a few minutes.
@st.cache_resource # Caches the model so it doesn't reload every time
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()
stop_words = set(stopwords.words('english'))


# --- 3. Define AI Functions ---
def generate_summary(text):
    # Adjust max_length and min_length for desired summary length
    summary_result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary_result[0]['summary_text']

def extract_keywords(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    # Filter out stopwords and non-alphabetic words
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    # Count word occurrences
    word_counts = Counter(filtered_words)
    # Return the top 10 most common words
    return [word for word, count in word_counts.most_common(10)]


# --- 4. Build Streamlit Web Interface ---
st.set_page_config(page_title="AI Text Summarizer & Keyword Extractor", layout="centered")

st.title("📄 AI Text Summarizer & Keyword Extractor")
st.markdown("Easily get the gist and key points from any long text using AI.")

# Input text area
text_input = st.text_area(
    "Paste your text here:",
    height=250,
    placeholder="e.g., A long article, research paper, or lecture notes..."
)

# Button to trigger processing
if st.button("✨ Summarize & Extract Keywords"):
    if text_input:
        with st.spinner("Processing your text..."):
            try:
                summary = generate_summary(text_input)
                keywords = extract_keywords(text_input)

                st.subheader("📝 Summary:")
                st.success(summary) # Green box for success

                st.subheader("🔑 Keywords:")
                # Display keywords as clickable badges (basic)
                st.info(", ".join(keywords)) # Blue box for info

            except Exception as e:
                st.error(f"An error occurred: {e}. Please try again or with a different text.")
                st.info("Large texts might sometimes cause issues, try shorter ones.")
    else:
        st.warning("Please enter some text to summarize.")

st.markdown("---")
st.markdown("Powered by Hugging Face Transformers & NLTK.")
st.markdown("Created for the 3MTT Knowledge Showcase.")
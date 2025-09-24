import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl','rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

#define sentiment prediction function
def predict_sentiment(text,model,vectorizer,stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
def initialize_scraper():
    return Nitter(instances='https://xcancel.com')

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# st.title("twitter Analysis App")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

# Main app logic
def main():
    st.title("Twitter Sentiment Analysis")

    # Load stopwords, model, vectorizer, and scraper only once
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    #scraper = initialize_scraper()
    #scraper = Nitter(log_level=1, skip_instance_check=False)


    # User input: either text input or Twitter username
    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        st.write(f"username: {username}")
        if st.button("Fetch Tweets"):
            nitter_instances = [
                    "https://nitter.net",
                    "https://nitter.privacydev.net",
                    "https://nitter.1d4.us",
                    "https://nitter.pussthecat.org",
                    "https://nitter.moomoo.me"
                ]
            scraper = Nitter(instances=nitter_instances,log_level=1,skip_instance_check=False) 
            tweets_data = scraper.get_tweets("@JeffBezos", mode='user')
            #tweets_data = scraper.get_tweets(username, mode='user', number=5)
            print(tweets_data)
            st.write(tweets_data)
            if 'tweets' in tweets_data:  # Check if the 'tweets' key exists
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']  # Access the text of the tweet
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)  # Predict sentiment of the tweet text
                    
                    # Create and display the colored card for the tweet
                    card_html = create_card(tweet_text, sentiment)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.write("No tweets found or an error occurred.")

if __name__ == "__main__":
    main()
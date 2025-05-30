import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompt template
summarize_prompt = PromptTemplate(
    input_variables=["article"],
    template="Summarize the following news article:\n\n{article}\n\nSummary:"
)

# Create the LLM chain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Function to extract news text
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ''.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url}: {e}"

# Streamlit UI
st.title("📰 News Summarizer using Gemini")
st.write("Enter a news article URL below to get a summary:")

user_url = st.text_input("URL", "")

if st.button("Summarize"):
    if user_url.strip():
        with st.spinner("Fetching and summarizing..."):
            article = extract_news(user_url)
            if article.startswith("Failed to fetch"):
                st.error(article)
            else:
                summary = summarize_chain.run(article=article)
                st.success("Summary generated!")
                st.subheader("Summary:")
                st.write(summary)
    else:
        st.warning("Please enter a valid URL.")

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup streamlit app
icon_url = "https://images.seeklogo.com/logo-png/61/1/deepseek-ai-icon-logo-png_seeklogo-611473.png"
st.set_page_config(page_title="DeepSeek AI Assistant", page_icon=icon_url)
st.markdown(
    f"""
    <h1 style='display: flex; align-items: center;'>
        <img src="{icon_url}" 
             style="width:40px; height:40px; margin-right:10px;" alt="DeepSeek AI Logo">
        DeepSeek AI Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

# App sidebar setting
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter Groq API Key: ", type="password")

# Create prompt
system_prompt = "You are a helpful assistant dedicated to answering all the user's queries."
prompt = ChatPromptTemplate.from_messages(
    [
        ( "system", system_prompt),
        ("human", "{query}"),
    ]
)

# Create output parser
output_parser = StrOutputParser()

if api_key:
    groq_api_key = api_key
    
    # Initialize model
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
    
    # Create chain
    chain = prompt | llm | output_parser
    
    # user input
    user_input = st.text_input("Ask you query: ")
    if user_input:
        with st.spinner("Thinking..."):
            response = chain.invoke({ "query": user_input })
            st.success(response)
else:
    st.error("Groq API Key is required to search your query.")


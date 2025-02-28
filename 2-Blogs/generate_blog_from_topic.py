import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Load env vars
load_dotenv()

# Initialize Llama3 model
llm = ChatGroq(model="llama3-70b-8192")


# Graph state
class State(TypedDict):
    topic: str
    title: str
    content: str


def generate_title(state: State):
    """Generates a blog title based on the topic in the state."""
    result = llm.invoke(f"Write a title for a blog about {state['topic']}")
    return {"title": result.content}


def generate_content(state: State):
    """Generates blog content based on the title in the state."""
    message = llm.invoke(f"Write blog content for the title {state['title']}")
    return {"content": message.content}


def build_workflow():
    """Builds and compiles the workflow."""
    # Build the workflow
    workflow = StateGraph(State)

    # Add nodes to workflow
    workflow.add_node("generate_blog_title", generate_title)
    workflow.add_node("generate_blog_content", generate_content)

    # Add edges to connect nodes
    workflow.add_edge(START, "generate_blog_title")
    workflow.add_edge("generate_blog_title", "generate_blog_content")
    workflow.add_edge("generate_blog_content", END)

    # Compile workflow
    return workflow.compile()


if __name__ == "__main__":
    # Streamlit app layout
    st.title("🤖 Blog Generator with LangGraph")

    # Get user input for the topic
    topic = st.text_input("Enter the topic of the blog")

    if st.button("Generate"):
        if topic:
            # Create the graph
            graph = build_workflow()
            
            # Save workflow
            graph.get_graph().draw_mermaid_png(output_file_path="generate_blog_from_topic.png")

            # Execute the workflow
            result = graph.invoke({"topic": topic})

            # Display title and content in Streamlit
            st.title(result["title"])
            st.write(result["content"])
        else:
            st.warning("Please enter a topic to generate the blog")

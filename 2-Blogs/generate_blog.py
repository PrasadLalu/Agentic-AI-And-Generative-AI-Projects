# LangGraph Blog Generation with Sequential Prompt Chaining
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load env vars
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Graph state
class State(TypedDict):
    topic: str
    title: str
    content: str
    
def generate_title(state: State):
    """
    Generates a blog title based on the topic in the state.

    Args:
        state: A dictionary containing the 'topic' key.

    Returns:
        A dictionary with the generated 'title'.
    """
    message = llm.invoke(f"Write a title for a blog about {state['topic']}")
    return {"title": message.content}


def generate_content(state: State):
    """
    Generates blog content based on the title in the state.

    Args:
        state: A dictionary containing the 'title' key.

    Returns:
        A dictionary with the generated 'content'.
    """
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
     # Create the graph
    graph = build_workflow()
    
    # Generate the Mermaid diagram and save as PNG
    graph.get_graph().draw_mermaid_png(output_file_path="generate_blog.png")

    # Execute the workflow with a sample topic
    result = graph.invoke({"topic": "Agentic AI"})
    print(result)
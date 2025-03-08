import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Load env vars
load_dotenv()

# Initialize Llama3 model
llm = ChatGroq(model="llama3-70b-8192")


# Initialize state
class State(TypedDict):
    query: str
    code: str
    test_cases: str
    test_cases_status: str


# Nodes
def generate_test_cases(state: State):
    query = state["query"]
    prompt = (
        f"Generate test cases for the following query:\n\n"
        f"Query: {query}\n\n"
        "Ensure the test cases cover edge cases, boundary values, and normal scenarios. "
        "Provide them in a structured format."
    )
    result = llm.invoke(prompt)
    return {"test_cases": result.content}


def generate_code(state: State):
    test_cases = state["test_cases"]
    prompt = f"Generate code that satisfies the following test cases:\n\n{test_cases}"
    result = llm.invoke(prompt)
    return {"code": result.content}


def review_test_cases(state: State):
    test_cases = state["test_cases"]
    code = state["code"]
    prompt = (
        f"Verify whether the following code passes all the given test cases:\n\n"
        f"Code:\n{code}\n\n"
        f"Test Cases:\n{test_cases}\n\n"
        "If all test cases pass, return 'success'. Otherwise, return 'failure' with an explanation."
    )
    result = llm.invoke(prompt)
    return {"test_cases_status": result.content}


def route_after_review(state: State):
    status = state["test_cases_status"]
    if "failure" in status.lower():
        return "generate_code"
    return END


# Build workflow
workflow_builder = StateGraph(State)

# Add nodes
workflow_builder.add_node("generate_test_cases", generate_test_cases)
workflow_builder.add_node("generate_code", generate_code)
workflow_builder.add_node("review_test_cases", review_test_cases)

# Add edges to connect nodes
workflow_builder.add_edge(START, "generate_test_cases")
workflow_builder.add_edge("generate_test_cases", "generate_code")
workflow_builder.add_edge("generate_code", "review_test_cases")
workflow_builder.add_conditional_edges(
    "review_test_cases",
    route_after_review,
    {"generate_code": "generate_code", END: END},
)
workflow_builder.add_edge("review_test_cases", END)

# Compile
workflow = workflow_builder.compile()

workflow.get_graph().draw_mermaid_png(output_file_path="micro_workflow.png")

# App setup
st.title("AI Code Generator")

# User input for query
input_query = st.text_input("Enter query to generate code:")

if st.button("Generate Code"):
    if input_query:
        state = workflow.invoke({"query": input_query})
        st.write("### Generated Code:")
        st.code(state["code"], language="python")
    else:
        st.warning("Please enter your query before generating code.")

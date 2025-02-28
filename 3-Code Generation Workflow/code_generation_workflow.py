import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Load env vars
load_dotenv()

# Initialize Llama3 model
llm = ChatGroq(model="llama3-8b-8192")

# Graph state
class State(TypedDict):
    code: str
    prompt: str
    testcases: str
    reviewed_code: str
    code_approved: str


# Define nodes
def generate_code(state: State):
    """LLM call to generate code from the given prompt."""
    result = llm.invoke(f"Write a programming code for {state['prompt']}")
    return {"code": result.content}


def write_testcases(state: State):
    """LLM call to write the test cases for the generated code."""
    result = llm.invoke(f"Write the test cases for the following code: {state['code']}")
    return {"testcases": result.content}


def code_reviewer(state: State):
    """LLM to review the generated code and provide efficiency suggestions."""
    result = llm.invoke(
        f"Make this code more efficient by giving suggestions with suggestions: {state['code']}"
    )
    result = llm.invoke(
        f"Act like a senior developer, review the code and suggest improvements for efficiency: {state['code']}"
    )
    return {"reviewed_code": result.content}


def code_evaluator(state: State):
    """LLM call to reviewing the code and providing approval status."""
    result = llm.invoke(
        f"Act like a project manager, review the following code and provide an approval status ('Pass' or 'Fail'): {state['reviewed_code']}"
    )
    return {"code_approved": result.content}


def check_approved(state: State):
    """Evaluate the approval status of the code"""
    if ( "approved" in state["code_approved"].lower() or "pass" in state["code_approved"].lower()):
        return "Pass"
    return "Fail"


# Build workflow
workflow_builder = StateGraph(State)

# Add nodes
workflow_builder.add_node("generate_code", generate_code)
workflow_builder.add_node("write_testcases", write_testcases)
workflow_builder.add_node("code_reviewer", code_reviewer)
workflow_builder.add_node("code_evaluator", code_evaluator)

# Add edges to connect nodes
workflow_builder.add_edge(START, "generate_code")
workflow_builder.add_edge("generate_code", "write_testcases")
workflow_builder.add_edge("write_testcases", "code_reviewer")

workflow_builder.add_conditional_edges(
    "code_evaluator", check_approved, {"Fail": "generate_code", "Pass": END}
)
workflow_builder.add_edge("code_reviewer", "code_evaluator")
workflow_builder.add_edge("code_evaluator", END)

# Compile the workflow
workflow = workflow_builder.compile()

workflow.get_graph().draw_mermaid_png(output_file_path="code_generation_workflow.png")

st.title("🤖 Code Generation and Review Workflow")

if __name__ == "__main__":
    user_prompt = st.text_input("Enter a prompt for code generation:")

    if st.button("Generate"):
        if user_prompt:
            # Invoke
            result = workflow.invoke({"prompt": user_prompt})

            # Check and display the generated code
            if "code" in result:
                st.markdown("#### Generated Code:")
                st.code(result["code"])
            else:
                st.warning("No code generated for the given prompt.")

            # Check and display the generated test cases
            if "testcases" in result:
                st.markdown("#### Test Cases:")
                st.code(result["testcases"])
            else:
                st.warning("No test cases found.")

            # Check and display the approval status of the code
            if "code_approved" in result:
                st.markdown("#### Approval Status:")
                st.write(result["code_approved"])
            else:
                st.warning("Code has not been approved.")
        else:            
            st.warning("Please give a prompt to generate code.")
            pass

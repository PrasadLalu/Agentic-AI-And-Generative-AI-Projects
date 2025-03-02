import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# Load env vars
load_dotenv()

# Initialize Llama3 model
llm = ChatGroq(model="llama3-70b-8192")


class Route(BaseModel):
    step: Literal["support", "tech", "marketing"] = Field(
        None, description="The next step in the routing process"
    )


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)


# Define state
class State(TypedDict):
    input: str
    output: str
    decision: str


# Define nodes
def handle_support_request(state: State):
    """LLM call for generating a solution for support based on the input."""
    result = llm.invoke(
        f"Please analyze provide a solution in a professional email support format:{state['input']}"
    )
    return {"output": result.content}


def handle_tech_request(state: State):
    """LLM call for generating a solution for tech based on the input."""
    result = llm.invoke(
        f"Please analyze provide a solution in a professional email tech format:{state['input']}"
    )
    return {"output": result.content}


def handle_marketing_request(state: State):
    """LLM call for generating a solution for marketing based on the input."""
    result = llm.invoke(
        f"Please analyze provide a solution in a professional email marketing format:{state['input']}"
    )
    return {"output": result.content}


def route_user_request(state: State):
    """Route the input to the appropriate node"""
    decision = router.invoke(
        [
            SystemMessage(
                content = "Route the input to marketing, tech, or support based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    if state["decision"] == "marketing":
        return "marketing_team"
    elif state["decision"] == "tech":
        return "tech_team"
    elif state["decision"] == "support":
        return "support_team"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("tech_team", handle_tech_request)
router_builder.add_node("support_team", handle_support_request)
router_builder.add_node("marketing_team", handle_marketing_request)
router_builder.add_node("user_request", route_user_request)

# Add edges to connect nodes
router_builder.add_edge(START, "user_request")
router_builder.add_conditional_edges(
    "user_request",
    route_decision,
    {
        "marketing_team": "marketing_team",
        "tech_team": "tech_team",
        "support_team": "support_team",
    },
)
router_builder.add_edge("marketing_team", END)
router_builder.add_edge("tech_team", END)
router_builder.add_edge("support_team", END)

# Compile workflow
router_workflow = router_builder.compile()

# Show the workflow
router_workflow.get_graph().draw_mermaid_png(output_file_path="langgraph_router.png")

# Setup streamlit app
st.title("🤖 LLM-Powered Query Handler")
st.subheader(
    "LLM-based routing and response generation for support, tech, and marketing inquiries"
)

# Sidebar setting
st.sidebar.title("App Workflow")
st.sidebar.image("langgraph_router.png")

# Text input field
query = st.text_input("What is your query?")

if st.button("Find Solution"):
    if query:
        # Invoke LLM Workflow
        state = router_workflow.invoke({"input": query})
        with st.spinner("Processing your query... 🎭"):
            state = router_workflow.invoke({"input": query})
        st.success(state["output"])
    else:
        st.warning("Please enter your query...")

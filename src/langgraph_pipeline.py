# =============================================================================
# LangGraph Agentic Pipeline (Stage 3)
# =============================================================================
# Stateful multi-node graph with reflection and re-retrieval

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from .retrieval import RetrievalEngine, RetrievedChunk


class AgentState(TypedDict):
    """State for LangGraph pipeline."""
    query: str
    retrieved_chunks: List[RetrievedChunk]
    context: str
    citations: List[dict]
    iteration: int
    should_continue: bool


MAX_ITERATIONS = 3


def retrieve_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant chunks."""
    query = state["query"]
    iteration = state.get("iteration", 0)
    
    print(f"\n[Retrieve] Iteration {iteration + 1}: '{query}'")
    
    engine = state.get("retrieval_engine")
    if engine:
        chunks = engine.retrieve(query)
    else:
        chunks = []
    
    state["retrieved_chunks"] = chunks
    state["iteration"] = iteration + 1
    
    print(f"[Retrieve] Retrieved {len(chunks)} chunks")
    
    return state


def reflect_node(state: AgentState) -> AgentState:
    """Node 2: Reflect on retrieval quality."""
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 1)
    
    # Check if we have enough relevant chunks
    if len(chunks) >= 3:
        state["should_continue"] = False
    elif iteration >= MAX_ITERATIONS:
        state["should_continue"] = False
    else:
        state["should_continue"] = True
    
    return state


def generate_node(state: AgentState) -> AgentState:
    """Node 3: Generate context for LLM."""
    chunks = state.get("retrieved_chunks", [])
    
    if not chunks:
        state["context"] = "No relevant information found."
        state["citations"] = []
        return state
    
    # Build context and citations with page numbers (FR-05)
    context_parts = []
    citations = []
    
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}]: {chunk.content}"
        )
        citations.append({
            "source": chunk.source,
            "page": chunk.page,
            "score": chunk.score
        })
    
    state["context"] = "\n\n".join(context_parts)
    state["citations"] = citations
    
    return state


def should_continue_edges(state: AgentState) -> str:
    """Decide whether to continue the loop."""
    if state.get("should_continue", False):
        return "continue"
    return "end"


def create_agentic_pipeline(retrieval_engine: RetrievalEngine):
    """Create LangGraph agentic pipeline."""
    
    def run_retrieval_node(state: AgentState) -> AgentState:
        state["retrieval_engine"] = retrieval_engine
        return retrieve_node(state)
    
    # Define the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("retrieve", run_retrieval_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("generate", generate_node)
    
    # Set entry point
    graph.set_entry_point("retrieve")
    
    # Add edges
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue_edges,
        {
            "continue": "retrieve",
            "end": "generate"
        }
    )
    graph.add_edge("generate", END)
    
    return graph.compile()


def run_agentic_pipeline(query: str, retrieval_engine: RetrievalEngine) -> dict:
    """Run the full agentic pipeline."""
    pipeline = create_agentic_pipeline(retrieval_engine)
    
    initial_state = {
        "query": query,
        "retrieved_chunks": [],
        "context": "",
        "citations": [],
        "iteration": 0,
        "should_continue": False,
        "retrieval_engine": retrieval_engine
    }
    
    result = pipeline.invoke(initial_state)
    
    return {
        "query": result.get("query", query),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "context": result.get("context", ""),
        "citations": result.get("citations", [])
    }
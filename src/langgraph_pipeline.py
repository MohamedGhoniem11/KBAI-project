# =============================================================================
# LangGraph Agentic Pipeline (Stage 3)
# =============================================================================
# Stateful multi-node graph with reflection and re-retrieval.
# Implements a blackboard architecture: nodes read/write shared AgentState.
# Loops: retrieve → reflect (enough? if not, re-retrieve) → generate.

from typing import TypedDict

from langgraph.graph import END, StateGraph  # LangGraph: stateful graph framework

from .retrieval import RetrievalEngine, RetrievedChunk


class AgentState(TypedDict):
    """Shared state (blackboard) for the LangGraph pipeline.
    Each node reads from and writes to this dictionary."""
    query: str                       # Original user question
    retrieved_chunks: list[RetrievedChunk]  # Evidence gathered so far
    context: str                     # Formatted context string (for LLM prompt)
    citations: list[dict]            # Citation metadata [source, page, score]
    iteration: int                   # Current loop iteration count
    should_continue: bool            # Decision: re-retrieve or proceed to generation


MAX_ITERATIONS = 3  # Maximum number of retrieve→reflect cycles


def retrieve_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant chunks via FAISS similarity search."""
    query = state["query"]
    iteration = state.get("iteration", 0)

    print(f"\n[Retrieve] Iteration {iteration + 1}: '{query}'")

    engine = state.get("retrieval_engine")
    chunks = engine.retrieve(query) if engine else []

    state["retrieved_chunks"] = chunks  # Store retrieved evidence on blackboard
    state["iteration"] = iteration + 1

    print(f"[Retrieve] Retrieved {len(chunks)} chunks")

    return state


def reflect_node(state: AgentState) -> AgentState:
    """Node 2: Reflect on retrieval quality — decide if we need more evidence.
    This is the backward-chaining step: "Do I have enough to answer the query?" """
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 1)

    # Continue if: we have < 3 chunks AND haven't exceeded max iterations
    if len(chunks) >= 3:
        state["should_continue"] = False  # Enough evidence → proceed to generation
    elif iteration >= MAX_ITERATIONS:
        state["should_continue"] = False  # Max retries reached → stop looping
    else:
        state["should_continue"] = True   # Not enough → re-retrieve

    return state


def generate_node(state: AgentState) -> AgentState:
    """Node 3: Package retrieved chunks into formatted context + citations for LLM.
    Note: This does NOT call the LLM — it just prepares the context string."""
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
    """Conditional edge: decide which node to route to next."""
    if state.get("should_continue", False):
        return "continue"  # Go back to retrieve_node
    return "end"           # Proceed to generate_node


def create_agentic_pipeline(retrieval_engine: RetrievalEngine):
    """Build and compile the LangGraph state graph.
    Graph structure:
      retrieve → reflect → (continue → retrieve | end → generate) → END
    """

    def run_retrieval_node(state: AgentState) -> AgentState:
        state["retrieval_engine"] = retrieval_engine
        return retrieve_node(state)

    # Define the graph with typed state
    graph = StateGraph(AgentState)

    # Add nodes (each is a function that transforms state)
    graph.add_node("retrieve", run_retrieval_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("generate", generate_node)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add edges
    graph.add_edge("retrieve", "reflect")              # Always go retrieve → reflect
    graph.add_conditional_edges(                         # Conditional: re-retrieve or generate?
        "reflect",
        should_continue_edges,
        {
            "continue": "retrieve",  # Loop back
            "end": "generate"        # Proceed to finish
        }
    )
    graph.add_edge("generate", END)  # Terminal

    return graph.compile()


def run_agentic_pipeline(query: str, retrieval_engine: RetrievalEngine) -> dict:
    """Run the full agentic pipeline and return results."""
    pipeline = create_agentic_pipeline(retrieval_engine)

    # Initial state — all fields empty except query
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

from src.workflows.graph_builder import BuildGraph
from langchain_core.messages import SystemMessage

if __name__ == '__main__':
    state = {
        'messages': [SystemMessage(content='Who is Dr APJ Abdul Kalam?')],
        'last_route': 'rag',
        'retry_count': 0
    }

    # Build and compile the graph
    app = BuildGraph(state=state).build_graph()

    # Run the stateful agent
    output = app.stream(state)

    for event in output:
        event["messages"][-1].pretty_print()

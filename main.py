from src.workflows.graph_builder import BuildGraph
from langchain_core.messages import SystemMessage


if __name__ == '__main__':

    state = {
        'messages': [SystemMessage(content="Tell me how GenAI is used in healthcare.")],
        'last_route': '',
        'retry_count': 0
    }

    # Build and compile the graph
    app = BuildGraph(state).build_graph()

    # Run the stateful agent
    output = app.stream(state)

    for event in output:
        print(event)

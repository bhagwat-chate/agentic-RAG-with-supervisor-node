from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from config.config_entity import ConfigEntity


class ValidationNode:

    def __init__(self):
        self.config = ConfigEntity()

    def route_validation(self, state: AgentState) -> str:
        return "pass" if state.get("validation_passed") else "fail"

    def validation_router(self, state: AgentState) -> AgentState:
        print(f"[ValidationNode.validation_router]: START: {state}")

        # Get retry count
        retry_count = state.get("retry_count", 0)

        # Custom validation logic here
        is_valid = self.some_validation(state)  # your actual logic
        state["validation_passed"] = is_valid

        # Update retry count if failed
        if not is_valid:
            retry_count += 1
            state["retry_count"] = retry_count
            if retry_count >= 3:  # 💥 Break loop after 3 retries
                print("Max retries exceeded.")
                state["validation_passed"] = True  # Treat as valid or trigger alternate exit
                state["messages"].append(SystemMessage(content="Max retries hit. Forcing pass."))
        else:
            state["retry_count"] = 0  # reset on success

        print(f"[ValidationNode.validation_router]: validation_passed={state['validation_passed']}, retry_count={retry_count}")
        return state

    def some_validation(self, state: AgentState) -> bool:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, SystemMessage):
            return False
        response = last_msg.content

        # Example: basic sanity checks
        return (
                len(response.strip()) > 20 and
                not response.strip().endswith("...") and
                "error" not in response.lower()
        )

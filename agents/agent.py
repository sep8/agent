import json
import yaml
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel
from agents.callback_manager import CallbackManager
from pathlib import Path
from agents.schema import AgentAction, AgentFinish, BaseTool, Callbacks
from models.chat_model import ChatModel


class BaseSingleActionAgent(BaseModel):
    """Base Agent class."""

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    # @abstractmethod
    # async def aplan(
    #     self,
    #     intermediate_steps: List[Tuple[AgentAction, str]],
    #     callbacks: Callbacks = None,
    #     **kwargs: Any,
    # ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        else:
            raise ValueError(
                f"Got unsupported early_stopping_method `{early_stopping_method}`"
            )

    @classmethod
    def from_llm_and_tools(
        cls,
        model: ChatModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        _type = self._agent_type
        _dict["_type"] = _type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        .. code-block:: python

            # If working with agent executor
            agent.agent.save(file_path="path/agent.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        return {}

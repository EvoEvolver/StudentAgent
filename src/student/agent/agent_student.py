import os
from typing import Dict

from .agent import Agent
from .agent_memory_helper import MemoryHelperAgent
from .memory import Memory
from .tools.tools import Tool


class StudentAgent(Agent):
    name = "StudentAgent"

    def __init__(
        self,
        tools: Dict[str, Tool] = {},
        cache=None,
        expensive=None,
        version=None,
        provider="openai",
        verbose=False,
        active_learning=True,
        memory_path=None,
    ):
        super().__init__(
            tools=tools,
            cache=cache,
            expensive=expensive,
            version=version,
            provider=provider,
            verbose=verbose,
        )

        self.active_learning = active_learning
        self.initialize_memory(memory_path)

    def initialize_memory(self, memory_path):
        self.memory_agent = MemoryHelperAgent(
            provider="openai",
            verbose=self.verbose,
            expensive=False,
            cache=True,
            logger=self.logger,
        )
        memory = Memory._load_from_file(memory_path) if memory_path else Memory("root")
        memory.set_helper_agent(self.memory_agent)
        self.memory = memory
        self.memory_path = memory_path

    def get_memory_agent(self):
        return self.memory_agent

    def get_memory(self):
        return self.memory

    def memory_size(self):
        return self.get_memory().get_memory_size()

    def get_tool_mask(self):
        mask = []
        if self.active_learning is False:
            mask.append("learning")
        return mask

    def _build_prompt(self, dir, version) -> str:
        # Reads the prompt file and returns it as a string.
        here = os.path.dirname(__file__)
        base_dir = os.path.join(here, "prompts", "system")

        path = os.path.join(base_dir, dir)
        path = os.path.join(path, f"{version}.xml")

        if not os.path.isfile(path):
            raise RuntimeError(f"Required prompt file missing: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()

        return text

    def save(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)

        super().save(folder_name)
        # self.get_memory_agent().save_memory(os.path.join(folder_name, "memory.parquet"))
        # self.get_memory_agent().save_conversation(os.path.join(folder_name, "conversation_memory.txt"))

    def load(self, folder_name):
        if len(self.conversation) == 0:
            return
        super().load(folder_name)

        # mem_path = os.path.join(folder_name, "memory.parquet")
        # mem_conv_path = os.path.join(folder_name, "conversation_memory.txt")
        # if os.path.exists(mem_path) and os.path.exists(mem_conv_path):
        #    try:
        #        self.get_memory_agent().load_memory(mem_path)
        #        self.get_memory_agent().load_conversation(mem_conv_path)
        #    except Exception as e:
        #        return e

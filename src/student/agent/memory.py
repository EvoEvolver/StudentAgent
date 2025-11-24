from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .agent_memory_helper import MemoryHelperAgent

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Memory:
    """Represents a single memory item in a hierarchical structure."""

    title: str
    creation_time: str
    content: Optional[str] = None  # Only for terminal memories
    children: List[Memory] = field(default_factory=list)  # Only for composite memories
    _helper_agent: Optional[MemoryHelperAgent] = field(default=None, repr=False)

    def __init__(
        self,
        title: str,
        content: Optional[str] = None,
        children: Optional[List[Memory]] = None,
        creation_time: Optional[str] = None,
        helper_agent: Optional[MemoryHelperAgent] = None,
    ):
        self.title = title
        self.creation_time = (
            creation_time if creation_time is not None else datetime.now().isoformat()
        )
        self.content = content
        self.children = children if children is not None else []
        self._helper_agent = helper_agent

    def __breadth__(self) -> int:
        """Calculate the size of the memory (number of terminal memories)."""
        if len(self.children) == 0:
            return 1
        else:
            return sum(child.__size__() for child in self.children)

    def __depth__(self) -> int:
        """Calculate the depth of the memory tree."""
        if len(self.children) == 0:
            return 1
        else:
            return 1 + max(child.__depth__() for child in self.children)

    def get_memory_size(self) -> Dict[str, int]:
        """Get the size and depth of the memory tree."""
        return {"size": self.__breadth__(), "depth": self.__depth__()}

    def _get_helper_agent(self) -> MemoryHelperAgent:
        """Get or create the helper agent for LLM operations."""
        if self._helper_agent is None:
            from .agent_memory_helper import MemoryHelperAgent

            self._helper_agent = MemoryHelperAgent(provider="openai", expensive=False)
        return self._helper_agent

    def set_helper_agent(self, agent: MemoryHelperAgent):
        """Set the helper agent for this memory and all children."""
        self._helper_agent = agent
        for child in self.children:
            child.set_helper_agent(agent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        children_dicts = [child.to_dict() for child in self.children]
        result = {
            "title": self.title,
            "creation_time": self.creation_time,
            "content": self.content,
            "children": children_dicts,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Create memory from dictionary."""
        children = [
            cls.from_dict(child_data) for child_data in data.get("children", [])
        ]
        return cls(
            title=data["title"],
            content=data.get("content"),
            children=children,
            creation_time=data.get("creation_time"),
        )

    def _save_to_file(self, storage_file):
        """Save memories to JSON file."""
        memories_dict = self.to_dict()
        with open(storage_file, "w", encoding="utf-8") as f:
            json.dump(memories_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def _load_from_file(cls, storage_file, create_if_missing=True) -> Memory:
        """Load memories from JSON file."""
        if not os.path.exists(storage_file):
            if create_if_missing:
                return cls(title="root")
            else:
                raise FileNotFoundError(
                    f"Memory storage file not found: {storage_file}"
                )
        with open(storage_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_children_by_title(self, title: str) -> Optional[Memory]:
        """Recursively search for a memory by title."""
        for child in self.children:
            if child.title == title:
                return child
        return None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_depth: int = 5,
        _current_depth: int = 0,
    ) -> List[Memory]:
        """
        Retrieve relevant memories based on a query.
        Uses LLM to filter titles and return matching contents.
        If composite memories are retrieved, recursively resolves them to return only terminal memories.

        Args:
            query: The search query
            top_k: Optional limit on number of results to return (applied before resolution)
            max_depth: Maximum recursion depth to prevent stack overflow
            _current_depth: Internal parameter tracking current recursion depth

        Returns:
            List of Memory objects (includes both composite and terminal memories)
        """
        if len(self.children) == 0 or _current_depth >= max_depth:
            return []

        agent = self._get_helper_agent()
        relevant_titles = agent.filter_relevant_titles(
            query, [mem.title for mem in self.children], top_k=top_k
        )

        relevant_memories = []
        for title in relevant_titles:
            for mem in self.children:
                if mem.title == title:
                    relevant_memories.append(mem)
                    break

        results = []
        for mem in relevant_memories:
            results.append(mem)
            if len(mem.children) > 0:
                # Composite memory - resolve recursively
                # Ensure child memories have access to the helper agent
                if mem._helper_agent is None:
                    mem._helper_agent = self._helper_agent
                resolved = mem.retrieve(
                    query,
                    top_k=None,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1,
                )
                results.extend(resolved)

        return results

    def learn(self, content: str) -> str:
        """
        Learn new information by generating a title and storing the content as a terminal memory.

        Args:
            content: The content to store

        Returns:
            The generated title
        """
        agent = self._get_helper_agent()
        existing_titles = [child.title for child in self.children]
        title = agent.generate_title(content, existing_titles=existing_titles)

        # Handle duplicate titles by appending a number
        original_title = title
        counter = 1
        while any(child.title == title for child in self.children):
            title = f"{original_title} ({counter})"
            counter += 1

        # Create new memory with same helper agent
        new_memory = Memory(
            title=title, content=content, helper_agent=self._helper_agent
        )
        self.children.append(new_memory)
        return title

    def learn_composite(self, children: List[str], title: Optional[str] = None) -> str:
        """
        Create a composite memory that groups related memories together.

        Args:
            children: List of memory titles to include in the composite
            title: Optional title for the composite memory. If not provided, will be auto-generated.

        Returns:
            The title of the created composite memory
        """
        # Find the Memory objects for the given titles
        child_memories = []
        for child_title in children:
            for child in self.children:
                if child.title == child_title:
                    child_memories.append(child)
                    break

        if not child_memories:
            raise ValueError("No valid child memories found with the provided titles")

        # Generate title if not provided
        if title is None:
            agent = self._get_helper_agent()
            title = agent.generate_composite_title(children)

            # Handle duplicate titles
            original_title = title
            counter = 1
            while any(child.title == title for child in self.children):
                title = f"{original_title} ({counter})"
                counter += 1

        # Create the composite memory with same helper agent
        composite_memory = Memory(
            title=title, children=child_memories, helper_agent=self._helper_agent
        )

        # Add to children
        self.children.append(composite_memory)

        # remove the individual memories that are now part of the composite
        for child_memory in child_memories:
            self.children.remove(child_memory)

        return title

    def merge_contents(self, content1: str, content2: str) -> str:
        """
        Merge two memory contents into a single coherent content.

        Args:
            content1: First memory content
            content2: Second memory content

        Returns:
            Merged content
        """
        agent = self._get_helper_agent()
        return agent.merge_contents(content1, content2)

    def check_similar_memories(self, max_pairs: int = 3) -> List[Dict[str, Any]]:
        """
        Analyze all memories and identify pairs of similar memories using LLM.

        Args:
            max_pairs: Maximum number of similar pairs to return (default: 3)

        Returns:
            List of similarity pairs, where each pair is a dict with:
            - "pair_id": Unique identifier for the pair
            - "description": Description of what makes these memories similar
            - "memory_titles": List of memory titles in this pair
        """
        if len(self.children) == 0:
            return []

        # Prepare memory list for LLM
        memory_list = []
        for idx, memory in enumerate(self.children, 1):
            if len(memory.children) == 0:
                content = memory.content or ""
                memory_list.append(
                    {
                        "index": idx,
                        "title": memory.title,
                        "content": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )
            else:
                memory_list.append(
                    {
                        "index": idx,
                        "title": memory.title,
                        "type": "composite",
                        "children": [child.title for child in memory.children],
                    }
                )

        agent = self._get_helper_agent()
        pairs = agent.check_similar_memories(memory_list, max_pairs=max_pairs)

        # Convert memory indices back to titles
        idx_to_title = {idx: child.title for idx, child in enumerate(self.children, 1)}

        formatted_pairs = []
        for pair in pairs:
            memory_indices = pair.get("memory_indices", [])
            memory_titles = [
                idx_to_title[idx] for idx in memory_indices if idx in idx_to_title
            ]

            formatted_pairs.append(
                {
                    "pair_id": pair.get("pair_id", 0),
                    "description": pair.get("description", ""),
                    "memory_titles": memory_titles,
                }
            )

        return formatted_pairs


if __name__ == "__main__":
    from student.agent.agent_memory_helper import MemoryHelperAgent

    print("Initializing MemoryHelperAgent...")
    helper_agent = MemoryHelperAgent(
        provider="openai",
        verbose=True,
    )

    if os.path.exists("memories.json"):
        memory = Memory._load_from_file("memories.json")
        memory.set_helper_agent(helper_agent)  # Set agent for loaded memory
    else:
        # Example usage
        memory = Memory("root", helper_agent=helper_agent)

        # Learn some terminal memories
        print("\nLearning terminal memories...")
        title1 = memory.learn(
            "Python is a high-level programming language known for its simplicity and readability."
        )
        print(f"Stored terminal memory: {title1}")

        title2 = memory.learn(
            "Machine learning is a subset of AI that enables systems to learn from data."
        )
        print(f"Stored terminal memory: {title2}")

        title3 = memory.learn(
            "Neural networks are computing systems inspired by biological neural networks."
        )
        print(f"Stored terminal memory: {title3}")

        title4 = memory.learn(
            "The Eiffel Tower is located in Paris, France and was completed in 1889."
        )
        print(f"Stored terminal memory: {title4}")

        # Create a composite memory that pairs related memories
        print("\nCreating composite memory...")
        composite_title = memory.learn_composite(
            children=[title2, title3], title="AI and Machine Learning Concepts"
        )
        print(f"Stored composite memory: {composite_title}")

        memory._save_to_file("memories.json")
        print("\nMemories saved to 'memories.json'.")

    # Retrieve information - composite memories are automatically resolved
    print("\nRetrieving information about AI (will resolve composite memory)...")
    results = memory.retrieve("Tell me about AI and machine learning")
    for result in results:
        print(f"\nTitle: {result.title}")
        print(f"Content: {result.content if result.content else '[Composite Memory]'}")

    print("\n" + "=" * 50)
    print("Note: The composite memory was automatically resolved to terminal memories!")

    # Check for similar memories
    print("\n" + "=" * 50)
    print("Checking for similar memories...")
    similar_pairs = memory.check_similar_memories()
    print(f"\nFound {len(similar_pairs)} pairs of similar memories:")
    for pair in similar_pairs:
        print(f"\nGroup {pair['pair_id']}: {pair['description']}")
        print(f"  Memories: {', '.join(pair['memory_titles'])}")

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MemoryType(Enum):
    """Type of memory item."""
    TERMINAL = "terminal"  # Contains actual content
    COMPOSITE = "composite"  # Contains references to other memories


@dataclass
class Memory:
    """Represents a single memory item."""
    title: str
    memory_type: MemoryType
    creation_time: str
    content: Optional[str] = None  # Only for terminal memories
    references: List[str] = field(default_factory=list)  # Only for composite memories

    @classmethod
    def create_terminal(cls, title: str, content: str) -> "Memory":
        """Create a new terminal memory with current timestamp."""
        return cls(
            title=title,
            memory_type=MemoryType.TERMINAL,
            content=content,
            creation_time=datetime.now().isoformat()
        )

    @classmethod
    def create_composite(cls, title: str, references: List[str]) -> "Memory":
        """Create a new composite memory that references other memories."""
        return cls(
            title=title,
            memory_type=MemoryType.COMPOSITE,
            references=references,
            creation_time=datetime.now().isoformat()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        result = {
            "title": self.title,
            "memory_type": self.memory_type.value,
            "creation_time": self.creation_time
        }
        if self.memory_type == MemoryType.TERMINAL:
            result["content"] = self.content
        else:
            result["references"] = self.references
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary."""
        memory_type = MemoryType(data["memory_type"])
        return cls(
            title=data["title"],
            memory_type=memory_type,
            creation_time=data["creation_time"],
            content=data.get("content"),
            references=data.get("references", [])
        )


class AgentMemoryV2:
    """
    Agent memory system that stores information as title-content pairs.
    Uses OpenAI LLM to generate titles when learning and filter relevant titles when retrieving.
    """

    def __init__(
            self,
            model: str = "gpt-5-nano",
            storage_file: Optional[str] = None
    ):
        """
        Initialize the memory system.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for title generation and filtering
            storage_file: Optional path to JSON file for persistent storage
        """
        self.client = OpenAI()
        self.model = model
        self.storage_file = storage_file
        self.memories: Dict[str, Memory] = {}  # title -> Memory object

        # Load from file if it exists
        if storage_file and os.path.exists(storage_file):
            self._load_from_file()

    def learn(self, content: str) -> str:
        """
        Learn new information by generating a title and storing the content as a terminal memory.

        Args:
            content: The content to store

        Returns:
            The generated title
        """
        title = self._generate_title(content)
        self.memories[title] = Memory.create_terminal(title=title, content=content)

        # Save to file if configured
        if self.storage_file:
            self._save_to_file()

        return title

    def learn_composite(self, references: List[str], title: Optional[str] = None) -> str:
        """
        Learn a composite memory that groups other memories together.

        Args:
            references: List of memory titles to reference
            title: Optional custom title. If not provided, will generate one based on referenced memories.

        Returns:
            The generated or provided title
        """
        # Validate that all references exist
        missing_refs = [ref for ref in references if ref not in self.memories]
        if missing_refs:
            raise ValueError(f"Referenced memories not found: {missing_refs}")

        # Generate title if not provided
        if title is None:
            # Create a summary of the referenced memories for title generation
            ref_titles = [ref for ref in references]
            summary = "Composite memory of: " + ", ".join(ref_titles[:3])
            if len(ref_titles) > 3:
                summary += f" and {len(ref_titles) - 3} more"
            title = self._generate_title(summary)

        self.memories[title] = Memory.create_composite(title=title, references=references)

        # Save to file if configured
        if self.storage_file:
            self._save_to_file()

        return title

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query.
        Uses LLM to filter titles and return matching contents.
        If composite memories are retrieved, recursively resolves them to return only terminal memories.

        Args:
            query: The search query
            top_k: Optional limit on number of results to return (applied before resolution)

        Returns:
            List of dicts with 'title', 'content', and 'creation_time' keys (only terminal memories)
        """
        if not self.memories:
            return []

        relevant_titles = self._filter_relevant_titles(query, list(self.memories.keys()))

        # Apply top_k limit if specified (before resolution)
        if top_k:
            relevant_titles = relevant_titles[:top_k]

        # Recursively resolve composite memories to get only terminal memories
        terminal_memories = self._resolve_to_terminal(relevant_titles)

        # Convert to dict format
        results = [memory.to_dict() for memory in terminal_memories]

        return results

    def _resolve_to_terminal(self, titles: List[str]) -> List[Memory]:
        """
        Recursively resolve a list of memory titles to terminal memories only.

        Args:
            titles: List of memory titles to resolve

        Returns:
            List of terminal Memory objects
        """
        terminal_memories = []
        visited = set()  # Track visited titles to avoid infinite loops

        def resolve_recursive(title: str):
            # Avoid infinite loops
            if title in visited:
                return
            visited.add(title)

            # Check if memory exists
            if title not in self.memories:
                return

            memory = self.memories[title]

            if memory.memory_type == MemoryType.TERMINAL:
                # Add terminal memory to results
                terminal_memories.append(memory)
            elif memory.memory_type == MemoryType.COMPOSITE:
                # Recursively resolve referenced memories
                for ref_title in memory.references:
                    resolve_recursive(ref_title)

        # Resolve each title
        for title in titles:
            resolve_recursive(title)

        return terminal_memories

    def _generate_title(self, content: str) -> str:
        """
        Use LLM to generate a concise title for the content.
        """
        prompt = f"""Generate a concise, descriptive title (max 10 words) for the following content.
The title should capture the key topic or information.
Only return the title, nothing else.

Content:
{content}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise titles."},
                {"role": "user", "content": prompt}
            ]
        )

        title = response.choices[0].message.content.strip()

        # Handle duplicate titles by appending a number
        original_title = title
        counter = 1
        while title in self.memories:
            title = f"{original_title} ({counter})"
            counter += 1

        return title

    def _filter_relevant_titles(self, query: str, titles: List[str]) -> List[str]:
        """
        Use LLM to filter and rank titles by relevance to the query.
        """
        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(titles)])

        prompt = f"""Given the following query and list of titles, select the titles that are relevant to the query.
Return ONLY the numbers of the relevant titles, ranked by relevance (most relevant first).
Format: comma-separated numbers (e.g., "3,1,7")
If no titles are relevant, return "NONE".

Query: {query}

Titles:
{titles_text}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that filters information by relevance."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content.strip()

        if result == "NONE":
            return []

        try:
            # Parse the returned indices
            indices = [int(x.strip()) - 1 for x in result.split(",")]
            # Validate indices and return corresponding titles
            relevant_titles = [titles[i] for i in indices if 0 <= i < len(titles)]
            return relevant_titles
        except (ValueError, IndexError):
            # If parsing fails, return empty list
            return []

    def _save_to_file(self):
        """Save memories to JSON file."""
        memories_dict = {title: memory.to_dict() for title, memory in self.memories.items()}
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(memories_dict, f, indent=2, ensure_ascii=False)

    def _load_from_file(self):
        """Load memories from JSON file."""
        with open(self.storage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.memories = {title: Memory.from_dict(mem_data) for title, mem_data in data.items()}

    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        if self.storage_file and os.path.exists(self.storage_file):
            self._save_to_file()

    def get_all_memories(self) -> Dict[str, Memory]:
        """Get all stored memories as Memory objects."""
        return self.memories.copy()

    def delete_memory(self, title: str) -> bool:
        """
        Delete a specific memory by title.

        Returns:
            True if deleted, False if title not found
        """
        if title in self.memories:
            del self.memories[title]
            if self.storage_file:
                self._save_to_file()
            return True
        return False


if __name__ == "__main__":
    # Example usage
    memory = AgentMemoryV2(storage_file="memory_storage.mem.json")

    # Learn some terminal memories
    print("Learning terminal memories...")
    title1 = memory.learn("Python is a high-level programming language known for its simplicity and readability.")
    print(f"Stored terminal memory: {title1}")

    title2 = memory.learn("Machine learning is a subset of AI that enables systems to learn from data.")
    print(f"Stored terminal memory: {title2}")

    title3 = memory.learn("Neural networks are computing systems inspired by biological neural networks.")
    print(f"Stored terminal memory: {title3}")

    title4 = memory.learn("The Eiffel Tower is located in Paris, France and was completed in 1889.")
    print(f"Stored terminal memory: {title4}")

    # Create a composite memory that groups related memories
    print("\nCreating composite memory...")
    composite_title = memory.learn_composite(
        references=[title2, title3],
        title="AI and Machine Learning Concepts"
    )
    print(f"Stored composite memory: {composite_title}")

    # Retrieve information - composite memories are automatically resolved
    print("\nRetrieving information about AI (will resolve composite memory)...")
    results = memory.retrieve("Tell me about AI and machine learning")
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"Content: {result['content']}")
        print(f"Created: {result['creation_time']}")

    print("\n" + "=" * 50)
    print("Note: The composite memory was automatically resolved to terminal memories!")

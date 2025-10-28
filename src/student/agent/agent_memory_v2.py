import json
import os
import base64
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

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
        self.memories: Dict[str, str] = {}  # title -> content

        # Load from file if it exists
        if storage_file and os.path.exists(storage_file):
            self._load_from_file()

    def learn(self, content: str) -> str:
        """
        Learn new information by generating a title and storing the content.

        Args:
            content: The content to store

        Returns:
            The generated title
        """
        title = self._generate_title(content)
        self.memories[title] = content

        # Save to file if configured
        if self.storage_file:
            self._save_to_file()

        return title

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve relevant memories based on a query.
        Uses LLM to filter titles and return matching contents.

        Args:
            query: The search query
            top_k: Optional limit on number of results to return

        Returns:
            List of dicts with 'title' and 'content' keys
        """
        if not self.memories:
            return []

        relevant_titles = self._filter_relevant_titles(query, list(self.memories.keys()))

        # Apply top_k limit if specified
        if top_k:
            relevant_titles = relevant_titles[:top_k]

        results = [
            {"title": title, "content": self.memories[title]}
            for title in relevant_titles
        ]

        return results

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
        titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])

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
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)

    def _load_from_file(self):
        """Load memories from JSON file."""
        with open(self.storage_file, 'r', encoding='utf-8') as f:
            self.memories = json.load(f)

    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        if self.storage_file and os.path.exists(self.storage_file):
            self._save_to_file()

    def get_all_memories(self) -> Dict[str, str]:
        """Get all stored memories."""
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

    # Learn some information
    print("Learning information...")
    title1 = memory.learn("Python is a high-level programming language known for its simplicity and readability.")
    print(f"Stored with title: {title1}")

    title2 = memory.learn("Machine learning is a subset of AI that enables systems to learn from data.")
    print(f"Stored with title: {title2}")

    title3 = memory.learn("The Eiffel Tower is located in Paris, France and was completed in 1889.")
    print(f"Stored with title: {title3}")

    # Retrieve information
    print("\nRetrieving information about programming...")
    results = memory.retrieve("Tell me about programming languages")
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"Content: {result['content']}")

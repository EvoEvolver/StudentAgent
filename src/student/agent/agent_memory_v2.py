from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


LLM_MODEL: str = "gpt-5-nano"

class Memory:
    """Represents a single memory item."""
    title: str
    creation_time: str
    content: Optional[str] = None  # Only for terminal memories
    children: List[Memory] = field(default_factory=list)  # Only for composite memories
    
    def __init__(
            self,
            title: str,
            content: Optional[str] = None,
            children: Optional[List[Memory]] = None,
            creation_time: Optional[str] = None
    ):
        self.title = title
        self.creation_time = creation_time if creation_time is not None else datetime.now().isoformat()
        self.content = content
        self.children = children if children is not None else []

    @classmethod
    def create_terminal(cls, title: str, content: str) -> Memory:
        """Create a new terminal memory with current timestamp."""
        return Memory(
            title=title,
            content=content,
        )

    @classmethod
    def create_composite(cls, title: str, children: List[Memory]) -> Memory:
        """Create a new composite memory that children other memories."""
        return cls(
            title=title,
            children=children,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        result = {
            "title": self.title,
            "creation_time": self.creation_time,
            "content": self.content,
            "children": self.children
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Create memory from dictionary."""
        return cls(
            title=data["title"],
            creation_time=data["creation_time"],
            content=data.get("content"),
            children=data.get("children", [])
        )


    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Memory]:
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
        if len(self.children) == 0:
            return []

        relevant_titles = _filter_relevant_titles(query, [mem.title for mem in self.children])

        # Apply top_k limit if specified (before resolution)
        if top_k:
            relevant_titles = relevant_titles[:top_k]

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
                resolved = mem.retrieve(query)
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
        title = self._generate_title(content)
        self.children.append(Memory.create_terminal(title=title, content=content))

        return title

    def learn_composite(self, children: List[str], title: Optional[str] = None) -> str:
        """
        Learn a composite memory that pairs other memories together.

        Args:
            children: List of memory titles to reference
            title: Optional custom title. If not provided, will generate one based on referenced memories.

        Returns:
            The generated or provided title
        """
        # Validate that all children exist
        missing_refs = [ref for ref in children if ref not in self.memories]
        if missing_refs:
            raise ValueError(f"Referenced memories not found: {missing_refs}")

        # Generate title if not provided
        if title is None:
            # Create a summary of the referenced memories for title generation
            ref_titles = [ref for ref in children]
            summary = "Composite memory of: " + ", ".join(ref_titles[:3])
            if len(ref_titles) > 3:
                summary += f" and {len(ref_titles) - 3} more"
            title = self._generate_title(summary)

        self.memories[title] = Memory.create_composite(title=title, children=children)

        return title

   

    def _generate_title(self, content: str) -> str:
        """
        Use LLM to generate a concise title for the content.
        """
        prompt = f"""Generate a concise, descriptive title (max 10 words) for the following content.
The title should capture the key topic or information.
Only return the title, nothing else.

Content:
{content}"""

        response = OpenAI().chat.completions.create(
            model=LLM_MODEL,
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


    def _merge_contents(self, content1: str, content2: str) -> str:
        """
        Use LLM to merge two memory contents into a single coherent content.

        Args:
            content1: First memory content
            content2: Second memory content

        Returns:
            Merged content
        """
        prompt = f"""Merge the following two pieces of information into a single coherent text.
Preserve all important details from both pieces while eliminating redundancy.
Keep the merged content concise and well-organized.

Content 1:
{content1}

Content 2:
{content2}

Return ONLY the merged content, nothing else."""

        response = OpenAI().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that merges information efficiently."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    def _generate_composite_title(self, referenced_titles: List[str]) -> str:
        """
        Use LLM to generate a title for a composite memory based on referenced memory titles.

        Args:
            referenced_titles: List of memory titles being referenced

        Returns:
            Generated title for the composite memory
        """
        titles_text = "\n".join([f"- {title}" for title in referenced_titles])

        prompt = f"""Generate a concise, descriptive title (max 10 words) that captures the common theme or topic of these related memories:

{titles_text}

Return ONLY the title, nothing else."""

        response = OpenAI().chat.completions.create(
            model=LLM_MODEL,
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

    def check_similar_memories(self) -> List[Dict[str, Any]]:
        """
        Analyze all memories and identify pairs of similar memories using LLM.

        Returns:
            List of similarity pairs, where each pair is a dict with:
            - "pair_id": Unique identifier for the pair
            - "description": Description of what makes these memories similar
            - "memory_titles": List of memory titles in this pair
        """
        if not self.memories:
            return []

        # Prepare memory list for LLM
        memory_list = []
        for idx, (title, memory) in enumerate(self.memories.items(), 1):
            if len(memory.children) == 0:
                memory_list.append({
                    "index": idx,
                    "title": title,
                    "content": memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                })
            else:
                memory_list.append({
                    "index": idx,
                    "title": title,
                    "type": "composite",
                    "children": memory.children
                })

        memories_text = json.dumps(memory_list, indent=2, ensure_ascii=False)

        prompt = f"""Analyze the following memories and identify pairs of similar or related memories.
Group memories should share common themes, topics, or concepts.

Memories:
{memories_text}

Return your response as a JSON array of pairs from the most similar to less similar pairs. 
Return at most 3 pairs.
Each pair should have:
- "pair_id": A number starting from 1
- "description": A brief description of what makes these memories similar
- "memory_indices": An array of memory indices that belong to this pair

If a memory doesn't fit into any pair, you can omit it or create a single-item pair if it's important.

Return ONLY the JSON array, no other text."""

        response = OpenAI().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that analyzes and pairs similar information. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content.strip()

        try:
            # Parse the JSON response
            result_data = json.loads(result_text)

            # Handle both {"pairs": [...]} and direct array formats
            if isinstance(result_data, dict) and "pairs" in result_data:
                pairs = result_data["pairs"]
            elif isinstance(result_data, list):
                pairs = result_data
            else:
                pairs = []

            # Convert memory indices back to titles
            idx_to_title = {idx: title for idx, title in enumerate(self.memories.keys(), 1)}

            formatted_pairs = []
            for pair in pairs:
                memory_indices = pair.get("memory_indices", [])
                memory_titles = [idx_to_title[idx] for idx in memory_indices if idx in idx_to_title]

                formatted_pairs.append({
                    "pair_id": pair.get("pair_id", 0),
                    "description": pair.get("description", ""),
                    "memory_titles": memory_titles
                })

            return formatted_pairs

        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, return empty list
            print(f"Error parsing LLM response: {e}")
            return []

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


def _filter_relevant_titles(query: str, titles: List[str]) -> List[str]:
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

    response = OpenAI().chat.completions.create(
        model=LLM_MODEL,
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


if __name__ == "__main__":
    # Example usage
    memory = AgentMemoryV2(storage_file="memory_storage.mem.json")

    res = memory.check_similar_memories()
    print(res)

if __name__ == "__main__1":
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

    # Create a composite memory that pairs related memories
    print("\nCreating composite memory...")
    composite_title = memory.learn_composite(
        children=[title2, title3],
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

    # Check for similar memories
    print("\n" + "=" * 50)
    print("Checking for similar memories...")
    similar_pairs = memory.check_similar_memories()
    print(f"\nFound {len(similar_pairs)} pairs of similar memories:")
    for pair in similar_pairs:
        print(f"\nGroup {pair['pair_id']}: {pair['description']}")
        print(f"  Memories: {', '.join(pair['memory_titles'])}")

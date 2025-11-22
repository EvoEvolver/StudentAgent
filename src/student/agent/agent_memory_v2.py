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


    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        children_dicts = [child.to_dict() for child in self.children]
        result = {
            "title": self.title,
            "creation_time": self.creation_time,
            "content": self.content,
            "children": children_dicts
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Create memory from dictionary."""
        children = [cls.from_dict(child_data) for child_data in data.get("children", [])]
        return cls(
            title=data["title"],
            content=data.get("content"),
            children=children,
            creation_time=data.get("creation_time")
        )

    def _save_to_file(self, storage_file):
        """Save memories to JSON file."""
        memories_dict = self.to_dict()
        with open(storage_file, 'w', encoding='utf-8') as f:
            json.dump(memories_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def _load_from_file(cls, storage_file, create_if_missing=True) -> Memory:
        """Load memories from JSON file."""
        if not os.path.exists(storage_file):
            if create_if_missing:
                return cls(title="root")
            else:
                raise FileNotFoundError(f"Memory storage file not found: {storage_file}")
        with open(storage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_children_by_title(self, title: str) -> Optional[Memory]:
        """Recursively search for a memory by title."""
        for child in self.children:
            if child.title == title:
                return child
        return None


    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Memory]:
        """
        Retrieve relevant memories based on a query.
        Uses LLM to filter titles and return matching contents.
        If composite memories are retrieved, recursively resolves them to return only terminal memories.

        Args:
            query: The search query
            top_k: Optional limit on number of results to return (applied before resolution)

        Returns:
            List of Memory objects (includes both composite and terminal memories)
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
        self.children.append(Memory(title=title, content=content))
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

        # Generate title if not provided
        if title is None:
            title = self._generate_composite_title(children)

        # Create the composite memory
        composite_memory = Memory(title=title, children=child_memories)

        # Add to children
        self.children.append(composite_memory)

        # remove the individual memories that are now part of the composite
        for child_memory in child_memories:
            self.children.remove(child_memory)

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
        while any(child.title == title for child in self.children):
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
        while any(child.title == title for child in self.children):
            title = f"{original_title} ({counter})"
            counter += 1

        return title



    def check_similar_memories(self) -> List[Dict[str, Any]]:
        """
        Analyze all memories and identify pairs of similar memories using LLM.

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
                memory_list.append({
                    "index": idx,
                    "title": memory.title,
                    "content": content[:200] + "..." if len(content) > 200 else content
                })
            else:
                memory_list.append({
                    "index": idx,
                    "title": memory.title,
                    "type": "composite",
                    "children": [child.title for child in memory.children]
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
            idx_to_title = {idx: child.title for idx, child in enumerate(self.children, 1)}

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
    if os.path.exists("memories.json"):
        memory = Memory._load_from_file("memories.json")
    else:
        # Example usage
        memory = Memory("root")

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

"""
MemoryHelperAgent: Handles all LLM operations for hierarchical memory system.

This agent provides specialized LLM operations for:
- Title generation
- Content merging
- Relevance filtering
- Similarity detection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent import Agent


class MemoryHelperAgent(Agent):
    """
    Specialized agent for handling LLM operations in hierarchical memory.
    Uses the pydantic-ai based Agent infrastructure for provider abstraction and logging.
    """

    def __init__(
        self,
        provider: str = "openai",
        expensive: bool = False,
        cache: bool = True,
        logger=None,
        verbose: bool = False,
    ):
        super().__init__(
            tools=[],  # No tools needed for memory helper
            system_prompt="You are a helpful assistant for memory management operations.",
            provider=provider,
            expensive=expensive,
            cache=cache,
            logger=logger,
            verbose=verbose,
        )

    def generate_title(self, content: str, existing_titles: List[str] = None) -> str:
        """
        Generate a concise title for memory content based on when it would be needed.

        Args:
            content: The memory content
            existing_titles: Optional list of existing titles to avoid duplicates

        Returns:
            Generated title (max 15 words) describing when the memory is needed
        """
        prompt = f"""Generate a concise title (max 15 words) that describes WHEN this memory would be needed or useful.
The title should describe the situation, question, or context where this information would be relevant.
Frame it as "When [situation/question/context]" or similar.
Only return the title, nothing else.

Content:
{content}"""

        if existing_titles:
            prompt += "\n\nNote: Avoid duplicating these existing titles:\n"
            prompt += "\n".join(f"- {title}" for title in existing_titles[-20:])

        title = self.single_run(prompt, expensive=self.expensive)
        return title.strip().strip('"').strip("'")

    def generate_composite_title(self, child_titles: List[str]) -> str:
        """
        Generate a title for a composite memory based on child memory titles.

        Args:
            child_titles: List of child memory titles

        Returns:
            Generated composite title (max 15 words)
        """
        titles_text = "\n".join([f"- {title}" for title in child_titles])

        prompt = f"""Generate a concise, descriptive title (max 15 words) that captures the common theme or topic of these related memories:

{titles_text}

Return ONLY the title, nothing else."""

        title = self.single_run(prompt, expensive=self.expensive)
        return title.strip().strip('"').strip("'")

    def merge_contents(self, content1: str, content2: str) -> str:
        """
        Merge two memory contents into a single coherent content.

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

        merged = self.single_run(prompt, expensive=self.expensive)
        return merged.strip()

    def filter_relevant_titles(
        self, query: str, titles: List[str], top_k: Optional[int] = None
    ) -> List[str]:
        """
        Filter and rank titles by relevance to the query.

        Args:
            query: Search query
            titles: List of titles to filter
            top_k: Optional limit on number of results

        Returns:
            List of relevant titles, ranked by relevance
        """
        if not titles:
            return []

        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(titles)])

        prompt = f"""Given the following query and list of titles, select the titles that are relevant to the query.
Return ONLY the numbers of the relevant titles, ranked by relevance (most relevant first).
Format: comma-separated numbers (e.g., "3,1,7")
If no titles are relevant, return "NONE".

Query: {query}

Titles:
{titles_text}"""

        result = self.single_run(prompt, expensive=self.expensive)
        result = result.strip()

        if result == "NONE" or not result:
            return []

        try:
            # Parse the returned indices
            indices = [int(x.strip()) - 1 for x in result.split(",") if x.strip()]
            # Validate indices and return corresponding titles
            relevant_titles = [titles[i] for i in indices if 0 <= i < len(titles)]

            # Apply top_k if specified
            if top_k is not None:
                relevant_titles = relevant_titles[:top_k]

            return relevant_titles
        except (ValueError, IndexError) as e:
            # If parsing fails, return empty list
            if self.verbose:
                print(f"Error parsing title filtering result: {e}")
                print(f"Raw result: {result}")
            return []

    def check_similar_memories(
        self, memory_list: List[Dict[str, Any]], max_pairs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Analyze memories and identify pairs of similar memories.

        Args:
            memory_list: List of memory dicts with 'index', 'title', 'content'/'children'
            max_pairs: Maximum number of pairs to return

        Returns:
            List of similarity pairs with:
            - "pair_id": Unique identifier for the pair
            - "description": Description of what makes these memories similar
            - "memory_indices": List of memory indices in this pair
        """
        if not memory_list:
            return []

        import json

        memories_text = json.dumps(memory_list, indent=2, ensure_ascii=False)

        prompt = f"""Analyze the following memories and identify pairs of similar or related memories.
Group memories should share common themes, topics, or concepts.

Memories:
{memories_text}

Return your response as a JSON object with a "pairs" key containing an array of pairs from the most similar to less similar pairs.
Return at most {max_pairs} pairs.
Each pair should have:
- "pair_id": A number starting from 1
- "description": A brief description of what makes these memories similar
- "memory_indices": An array of memory indices that belong to this pair

If a memory doesn't fit into any pair, you can omit it or create a single-item pair if it's important.

Return ONLY the JSON object, no other text."""

        result = self.single_run(prompt, expensive=self.expensive)

        try:
            # Parse the JSON response
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result

            # Handle both {"pairs": [...]} and direct array formats
            if isinstance(result_data, dict) and "pairs" in result_data:
                pairs = result_data["pairs"]
            elif isinstance(result_data, list):
                pairs = result_data
            else:
                pairs = []

            return pairs

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If parsing fails, return empty list
            if self.verbose:
                print(f"Error parsing similarity check result: {e}")
                print(f"Raw result: {result}")
            return []

    def extract_keywords(
        self, content: str, existing_keywords: List[str] = None, max_keywords: int = 5
    ) -> List[str]:
        """
        Extract relevant keywords from content.

        Args:
            content: Content to extract keywords from
            existing_keywords: Optional list of existing keywords to consider
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        prompt = f"""Extract {max_keywords} most relevant keywords or key phrases from the following content.
Return ONLY the keywords, one per line, nothing else.

Content:
{content}"""

        if existing_keywords:
            prompt += (
                "\n\nHere are some existing keywords in the system for reference:\n"
            )
            prompt += ", ".join(existing_keywords[:30])

        result = self.single_run(prompt, expensive=self.expensive)
        keywords = [
            kw.strip().strip("-").strip("*").strip()
            for kw in result.split("\n")
            if kw.strip()
        ]
        return keywords[:max_keywords]

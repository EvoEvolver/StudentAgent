from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import pandas as pd

from .agent import Agent
from .agent_memory import MemoryAgent
from typing import List, Dict
from mllm import Chat, get_embeddings


from .agent_memory import Ask

def memory_agent_tools(provider, memory_path):
    agent = RAGAgent(provider=provider, memory_path=memory_path)
    return {'agent': agent, 'tools' : [Ask(agent)]}


class RAGAgent(MemoryAgent):
    """
    Simple Retrieval-Augmented-Generation agent that

    • keeps a conversational “memory” as <text, embedding> rows  
    • stores / reloads that memory with pandas → Parquet  
    • retrieves the Top-k most similar memories for every user prompt  
    • prepends those snippets as context before delegating to `Agent.run()`
    """
    def run(self, prompt: str, max_iter: int = 15):

        context_snippets = self.retrieve_memory(prompt)
        context = '\n'.join(context_snippets) if context_snippets else '<<no context found>>'
        augmented_prompt +=f"Context:\n{context}\n\n"
        augmented_prompt += f"Question: {prompt}\n Assistant: "
        
        return self._run(augmented_prompt, max_iter)







class RAGMemory:
    # ------------------------------------------------------------------ #
    # Construction / persistence
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        memory_path: str,
        cache=None,
        expensive=None,
        dir=None,
        version=None,
        provider: str = "anthropic",
        verbose: bool = False,
        *,
        model_embed: str = "text-embedding-3-small",
        top_k: int = 3,
    ):
        super().__init__(tools=[], cache=cache, expensive=expensive,
                         dir=dir, version=version, provider=provider, verbose=verbose)

        self._memory_path = memory_path
        self.model_embed = model_embed
        self.top_k = top_k
        
        rag_prompt = "You are a helpful assistant. Answer strictly from the context if possible.\n\n"
        self.reset_system_prompt(rag_prompt)

        # Load or initialise an empty memory bank
        if os.path.isfile(memory_path):
            self._load_memory()
        else:
            os.makedirs(os.path.dirname(memory_path) or ".", exist_ok=True)
            self.memory: List[str] = []
            self._doc_vectors = np.zeros((0, 0), dtype=np.float32)
            self._doc_norms = np.zeros(0, dtype=np.float32)
            self._save_memory()


    def add_memory(self, text: str) -> None:
        """
        Store a new piece of text in memory and persist it.
        """
        self.memory.append(text)
        vec = self._embed_texts([text])[0]

        # First insert determines embedding dim
        if self._doc_vectors.size == 0:
            self._doc_vectors = vec.reshape(1, -1)
        else:
            self._doc_vectors = np.vstack([self._doc_vectors, vec])

        self._doc_norms = np.append(self._doc_norms, np.linalg.norm(vec))
        self._save_memory()

    def retrieve_memory(self, query: str | None = None) -> List[str]:
        """
        Return *all* memories if `query` is None, otherwise the Top-k most
        semantically similar snippets.
        """
        if query is None or len(self.memory) == 0:
            return self.memory

        q_vec = self._embed_texts([query])[0]
        q_norm = np.linalg.norm(q_vec)
        scores = (self._doc_vectors @ q_vec) / (self._doc_norms * q_norm + 1e-8)
        idx = np.argsort(scores)[-self.top_k :][::-1]
        return [self.memory[i] for i in idx]

    # The main entry-point that callers will use
    def run(self, prompt: str, max_iter: int = 15):

        context_snippets = self.retrieve_memory(prompt)
        context = '\n'.join(context_snippets) if context_snippets else '<<no context found>>'
        augmented_prompt +=f"Context:\n{context}\n\n"
        augmented_prompt += f"Question: {prompt}\n Assistant: "
        
        return self._run(augmented_prompt, max_iter)

    
    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        response = get_embeddings(texts)
        vectors = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
        return np.asarray(vectors, dtype=np.float32)

    # ---------- Parquet helpers --------------------------------------- #
    def _save_memory(self) -> None:
        df = pd.DataFrame(
            {
                "text": self.memory,
                "embedding": self._doc_vectors.astype(np.float32).tolist(),
            }
        )
        df.to_parquet(self._memory_path, index=False)

    def _load_memory(self) -> None:
        df = pd.read_parquet(self._memory_path)
        if {"text", "embedding"} - set(df.columns):
            raise ValueError(
                f"{self._memory_path} is missing required columns 'text' and 'embedding'."
            )
        self.memory = df["text"].tolist()
        self._doc_vectors = np.asarray(df["embedding"].tolist(), dtype=np.float32)
        self._doc_norms = np.linalg.norm(self._doc_vectors, axis=1)
import numpy as np
from typing import List, Dict
from student.agent.memory import Memory, MemoryNode

import pandas as pd
import json
import numpy as np
from typing import Dict, Set
import copy


class MemoryNodeRAG(MemoryNode):
    id : str
    keys : Set[str]
    embeddings : List[str]
    content : str    

    def __init__(self, input: str=""):
        super().__init__(content="", keys = [input])
    
    def _get_embedding_score(self, query: str, sensitivity=0.4):
        if query is None or not str(query).strip():
            # No meaningful query → zero score, avoids embedding call entirely
            return np.array([[0.0]])
        similarity = self._get_cosine_similarity([query], keys=None)
        similarity = similarity * (similarity > sensitivity)
        return similarity

    def get_score(self, query: str, sensitivity=0.4):
        sim = self._get_embedding_score(query, sensitivity)
        if sim.size == 0:
            return 0.0
        return float(sim[0][0])


    def __str__(self):
        return next(iter(self.keys))
    
    def __copy__(self):
        return copy.deepcopy(self)
    

    @classmethod
    def from_dict(cls, d):
        new_node = MemoryNodeRAG()  
        new_node._from_dict(d)
        return new_node

    def render_html(self) -> str:
        import html

        content_html = html.escape(next(iter(self.keys))).replace("\n", "<br>")

        return f"""\
    <style>
    .memory-node {{
        font-family: system-ui, sans-serif;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 1rem;
        margin: .5rem 0;
        max-width: 200px;
    }}
    .memory-node h3 {{ margin: 0 0 .5rem 0; font-size: 1.25rem; }}
    .memory-node ul {{ margin: .25rem 0 .75rem 1rem; }}
    .memory-node p {{ margin: 0; white-space: pre-wrap; }}
    </style>

    <div class="memory-node">
    <p>{content_html}</p>
    </div>
    """


class MemoryRAG(Memory):
    memory : Dict[str, MemoryNodeRAG]
    keywords : Set[str]

    def __init__(self):
        super().__init__()
        self.score_matching = False
    
    def add_from_dict(self, node_dict: Dict) -> None:
        node = MemoryNodeRAG()
        node._from_dict(node_dict)
        self.add(node)

    
    def _recall(self, query: str, max_recall=5, sensitivity=0.3, thres=0.3) -> Dict[str, float]:
        nodes = self.get_nodes()
        if len(nodes) == 0:
            return {}
        
        excited_nodes = {}
        scores = self.get_scores(nodes, query, sensitivity)
        
        top_k_indices = np.argsort(-scores)
        
        for i in range(min(max_recall, len(top_k_indices))):
            s = scores[top_k_indices[i]]
            if s <= thres:
                break

            node : MemoryNodeRAG = nodes[top_k_indices[i]]
            excited_nodes[node.id] = s
        return excited_nodes
    

    def recall(self, query: str, max_recall=5, sensitivity=0.3, thres=0.3) -> Dict[str, str]:
        excited_nodes = self._recall(query, max_recall=max_recall, sensitivity=sensitivity, thres=thres)
        out = {}
        for id, score in excited_nodes.items():
            node = self.get_node(id)
            out[id] = node.__str__() 
        return out

    def load_text(self, load_path):
        # load the memory from a file
        with open(load_path) as f:
            memory_list = json.load(f)
        for d in memory_list:
            try:
                node = MemoryNodeRAG().from_dict(d)
                self.memory[node.id] = node
            except ValueError as e:
                print(e, "for dictionary: ", d)
        self.load_keywords()

    def load(self, load_path: str, clear=False) -> None:
        df = pd.read_parquet(load_path)
        if clear is True:
            self.memory.clear()
        for _, row in df.iterrows():
            node = MemoryNodeRAG.from_dict(row.to_dict())
            self.memory[node.id] = node

        self.load_keywords()

    def update_keywords(self, keyword):
        pass
    
    def load_keywords(self):
        """
        Load keywords from the memory nodes into the keyword dictionary.
        This is useful after loading a memory from a file.
        """
        self.keywords = set()

    def load_from_memory(self, memory: Memory):
        if type(memory) == MemoryRAG:
            self.memory = memory.__copy__().memory
        else:
            for id, node in memory.memory.items():
                if len(node.content) > 0:
                    new_node = MemoryNodeRAG(input=node.content)
                    new_node.id = id
                    self.add(new_node)
            print("Memory converted to RAG memory")

    def __copy__(self):
        return copy.deepcopy(self)
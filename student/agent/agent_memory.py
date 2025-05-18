import json
import uuid
import numpy as np
from mllm import Chat, get_embeddings
from typing import Dict, List, Set

from .bm25_indexing import get_bm25_score


class MemoryNode:
    id : str
    keys : Set[str]
    embeddings : List[str]
    content : str
    # assocations : list[str] # list of keys associated to this node
    

    def __init__(self, content: str = "", keys: List[str] = []):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.keys = set()
        self.embeddings = []

        self.add_keys(keys)
        # self.associations = []

    def get_keys(self):
        return list(self.keys)

    def add_keys(self, new_keys: List[str]):
        assert isinstance(new_keys, List)
        for key in new_keys:
            assert isinstance(key, str)
            self.keys.add(key)
        return self.keys
    

    def remove_keys(self, rem_keys: Set[str]):
        assert isinstance(rem_keys, Set)
        for key in list(rem_keys):
            assert isinstance(key, str)
            self.keys.remove(key)
        return self.keys


    def set_embeddings(self):
        self.embeddings = get_embeddings(list(self.keys))


    #def update_associations(self, new_associations):
    #    self.assocations = new_associations

    '''
    def get_score(self, input_keys, sensitivity=0):
        input_keys_embeddings = np.array(get_embeddings(input_keys))
        embeddings = np.array(self.embeddings)

        embed_similarity = np.dot(embeddings, input_keys_embeddings.T)
        embed_similarity = embed_similarity * (embed_similarity > sensitivity)
        embed_similarity = np.sum(embed_similarity, axis=0)
        
        bm25__similarity = get_bm25_score(self.keys, input_keys)
        
        return embed_similarity + bm25__similarity
    '''

    def _get_embedding_score(self, query : List[str], keys: List[str]=None, sensitivity=0.4):
        q_emb = np.array(get_embeddings(query))
        
        if keys is None:
            keys = self.keys
            if self.embeddings == []:
                self.set_embeddings()
            k_emb = np.array(self.embeddings)
        else:
            k_emb = np.array(get_embeddings(keys))
        
        similarity = np.dot(q_emb, k_emb.T) # len(query) x len(keys)
        q = np.linalg.norm(q_emb, axis=1, keepdims=True)
        k = np.linalg.norm(k_emb, axis=1)

        norm = q * k + (q-k)**2 + 1
        similarity = 2 * similarity / norm
        '''
        for i in range(len(query)):
            q = np.linalg.norm(q_emb[i])
            for j in range(len(keys)):
                k = np.linalg.norm(k_emb[i])

                similarity[i][j] /= k * q +((q-k)**2 + 1)
                similarity[i][j] *= 2
        '''     

        similarity = similarity * (similarity > sensitivity) # filter out bad matches
        return similarity 


    def get_score(self, query : List[str], keys: List[str] = None, sensitivity=0.4):
        scores_emb = self._get_embedding_score(query, keys, sensitivity) # len(query) x len(keys)
        scores_max = np.max(scores_emb, axis=0) # max score per key # len(keys)
        score_final = np.mean(scores_max) # average over keys
        return score_final


    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "keys": list(self.keys),
        }

    def _from_dict(self, d):
        self.content = d["content"]
        self.keys = set(d["keys"])
        if "id" in d.keys():
            self.id = d["id"]
    
    @classmethod
    def from_dict(cls, d):
        new_node = MemoryNode()  
        new_node._from_dict(d)
        return new_node

    def __str__(self):
        return f"""
        <memory id="{self.id}">
            <stimuli>{", ".join(self.keys)}</stimuli>
            <content>{self.content}</content>
        </memory>
        """
    

    def render_html(self) -> str:
        """
        Return a readable HTML representation of a MemoryNode.
        `max_emb_len` controls how many chars of each embedding to show before adding an ellipsis.
        """
        from IPython.display import HTML
        import html

        keys_html = "".join(f"<li>{html.escape(k)}</li>" for k in sorted(self.keys))
        content_html = html.escape(self.content).replace("\n", "<br>")

        return HTML(f"""\
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
    <strong>Keys</strong>
    <ul>{keys_html}</ul>

    <strong>Content</strong>
    <p>{content_html}</p>
    </div>
    """)



class Memory:
    memory : Dict[str, MemoryNode]
    keywords : Set[str]

    def __init__(self):
        self.memory : Dict[str, MemoryNode] = {}
        self.keywords = set()
    
    def __size__(self) -> int:
        return len(self.memory.keys())
    
    '''
    def get_keywords(self) -> Set[str]:
        keywords : Set[str] = set()
        for node in self.get_nodes():
            for k in node.keys:
                keywords.add(k)
        return keywords
    '''

    def get_node(self, id):
        return self.memory.get(id)
    
    def delete_node(self, id):
        del self.memory[id]
        return

    def add(self, node: MemoryNode):
        self.memory[node.id] = node
        for k in node.keys:
            self.keywords.add(k)
    
    def add_from_dict(self, node_dict: Dict) -> None:
        node = MemoryNode()
        node._from_dict(node_dict)
        self.add(node)


    def get_nodes(self) -> List[MemoryNode]:
        nodes = []
        for node in self.memory.values():
            if len(node.content) > 0 and len(node.keys) > 0:
                nodes.append(node)
                node.set_embeddings()
        return nodes
    
    def get_scores(self, nodes, queries: List[str], sensitivity=0.01):
        scores = []
        for node in nodes:
            node_scores = node.get_score(queries, sensitivity=sensitivity)
            node_scores = np.array(node_scores)
            scores.append(node_scores)

        scores = np.array(scores) # m

        #score_summation_for_src = np.sum(scores, axis=0)

        #score_norm_factor_for_src = score_summation_for_src + (score_summation_for_src == 0.0)
        #scores = scores / score_norm_factor_for_src
        #scores = np.sum(scores, axis=1)
        return scores

    def _recall(self, queries: List[str], max_recall=5, sensitivity=0.3, thres=0.3) -> Dict[str, float]:
        nodes = self.get_nodes()
        if len(nodes) == 0:
            return {}
        
        excited_nodes = {}
        scores = self.get_scores(nodes, queries, sensitivity)
        
        top_k_indices = np.argsort(-scores)
        
        for i in range(min(max_recall, len(top_k_indices))):
            s = scores[top_k_indices[i]]
            if s <= thres:
                break

            node : MemoryNode = nodes[top_k_indices[i]]
            excited_nodes[node.id] = s
        
        return excited_nodes
    
    def recall(self, queries: List[str], max_recall=5, sensitivity=0.3, thres=0.3) -> Dict[str, str]:
        excited_nodes = self._recall(queries, max_recall=max_recall, sensitivity=sensitivity, thres=thres)
        out = {}
        for id, score in excited_nodes.items():
            node = self.get_node(id)
            out[id] = node.__str__() 
        return out


        
    
    def modify(self, id: str, new_stimuli: List[str] = None, new_content: str = None) -> None:
        node : MemoryNode = self.get_node(id)
        deleted=False

        if node is None:
            return None, None
        
        if new_stimuli is not None:
            node.remove_keys(node.keys)
            node.add_keys(new_stimuli)

        if new_content is not None:
            node.content = new_content
        
        if new_content is None and new_stimuli is None:
            self.delete_node(id)
            deleted=True

        return node, deleted


    def save(self, save_path):
        # save the memory to a file
        memory_list = []
        for node in self.memory.values():
            memory_list.append(node.to_dict())
        # save by json
        with open(save_path, "w") as f:
            json.dump(memory_list, f)

    def load(self, load_path):
        # load the memory from a file
        with open(load_path) as f:
            memory_list = json.load(f)
        for d in memory_list:
            node = MemoryNode().from_dict(d)
            self.memory[node.id] = node


    def render_html(self, *, max_emb_len: int = 12):
        """
        Render an entire Memory object, laying each MemoryNode side-by-side.

        Parameters
        ----------
        mem : Memory
            The Memory instance whose nodes you want to visualize.
        Returns
        -------
        IPython.display.HTML
            A single HTML object showing all nodes in a flexbox container.
        """
        from IPython.display import HTML
        import html
        node_snippets = [
            node.render_html().data
            for node in self.memory.values()
        ]

        combined_html = f"""\
    <style>
    /* Flex container to place nodes side-by-side and wrap nicely. */
    .memory-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }}
    </style>
    <div class="memory-container">
        {''.join(node_snippets)}
    </div>
    """
        return HTML(combined_html)
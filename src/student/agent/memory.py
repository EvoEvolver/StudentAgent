import pandas as pd
import json
import uuid
import numpy as np
from mllm import Chat, get_embeddings
from typing import Dict, List, Set, Iterable, Tuple
class MemoryNode:
    id : str
    keys : Set[str]
    embeddings : List[str]
    content : str    

    def __init__(self, content: str = "", keys: List[str] = []):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.keys = set()
        self.embeddings = []

        self.add_keys(keys)


    def check_keys(self):
        if len(self.keys) == 0:
            raise ValueError("MemoryNode must have at least one key of length > 0")
 
    def get_keys(self):
        return list(self.keys)

    def add_keys(self, new_keys: List[str]):
        assert isinstance(new_keys, List)
        for key in new_keys:
            assert isinstance(key, str)
            key = self.format_key(key)
            if len(key) > 0:    # avoid empty strings as key
                self.keys.add(key)
        return self.keys
    

    def remove_keys(self, rem_keys: Set[str], check=True):
        assert isinstance(rem_keys, Set)
        for key in list(rem_keys):
            assert isinstance(key, str)
            self.keys.remove(key)
        if check is True:
            self.check_keys()
        return self.keys

    def format_key(self, key):
        return key.strip(" \t\n")

    def clean_keys(self):
        for key in self.keys:
            if key and len(self.format_key(key)) == 0:
                self.keys.remove(key)
    

    def set_embeddings(self):
        if len(self.embeddings) == len(self.keys):
            return True
        
        self.clean_keys() # keys are not allowed to be empty!
        keys = list(self.keys) 

        if len(keys) > 0:
            self.embeddings = get_embeddings(keys)
            return True
        else: 
            return False


    def _get_cosine_similarity(self, query : List[str], keys: List[str]=None):
        q_emb = np.array(get_embeddings(query))
        
        if keys is None:
            keys = self.keys
            if self.embeddings == []:
                if self.set_embeddings() is False:
                    return np.array([0])

            k_emb = np.array(self.embeddings)
        else:
            k_emb = np.array(get_embeddings(keys))
        
        similarity = np.dot(q_emb, k_emb.T) # len(query) x len(keys)
        q = np.linalg.norm(q_emb, axis=1, keepdims=True)
        k = np.linalg.norm(k_emb, axis=1)

        norm = q * k + (q-k)**2 + 1            
        similarity = 2 * similarity / norm  # strictly equal to cosine similarity for normalized vectors!
        '''
        for i in range(len(query)):
            q = np.linalg.norm(q_emb[i])
            for j in range(len(keys)):
                k = np.linalg.norm(k_emb[i])

                similarity[i][j] /= k * q +((q-k)**2 + 1)
                similarity[i][j] *= 2
        '''     
        return similarity
    
    def _get_embedding_score(self, query : List[str], keys: List[str]=None, sensitivity=0.4):
        similarity = self._get_cosine_similarity(query, keys)
        similarity = similarity * (similarity > sensitivity) # filter out bad matches
        return similarity 


    def get_score(self, query : List[str], keys: List[str] = None, sensitivity=0.4):
        scores_emb = self._get_embedding_score(query, keys, sensitivity) # len(query) x len(keys)
        scores_max = np.max(scores_emb, axis=0) # max score per key # len(keys)
        score_final = np.mean(scores_max) # average over keys
        return score_final



    def _get_similarity_raw(self, query: List[str], keys: List[str] = None) -> np.ndarray:
        q_emb = np.array(get_embeddings(query))
        if keys is None:
            keys = self.keys
            if self.embeddings == []:
                if self.set_embeddings() is False:
                    return np.zeros((len(query), 0))
            k_emb = np.array(self.embeddings)
        else:
            k_emb = np.array(get_embeddings(keys))

        similarity = np.dot(q_emb, k_emb.T)                # (q, k)
        q = np.linalg.norm(q_emb, axis=1, keepdims=True)   # (q, 1)
        k = np.linalg.norm(k_emb, axis=1)                  # (k,)
        norm = q * k + (q - k) ** 2 + 1                    # broadcasts -> (q, k)
        similarity = 2 * similarity / norm                 # equals cosine if vectors are L2-normalised
        return similarity                                   # DO NOT threshold here


    def _scale_and_threshold(self, sim: np.ndarray, tau: float) -> np.ndarray:
        """
        Build \tilde S = max(0, (sim - tau)/(1 - tau)), clipped to [0, 1].
        Works well when sim <= 1 (cosine); clipping protects if sim can exceed 1.
        """
        if tau >= 1.0:
            raise ValueError("tau must be < 1.")
        S = (sim - tau) / (1.0 - tau)
        S = np.clip(S, 0.0, 1.0)   # zeros out <= tau; caps huge values
        return S


    def _greedy_max_weight_matching(self, W: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Greedy 1-to-1 matching on nonnegative weight matrix W (q x k).
        Sort edges by weight desc, take an edge if both endpoints are unused.
        """
        if W.size == 0:
            return []
        q, k = W.shape
        used_q = np.zeros(q, dtype=bool)
        used_k = np.zeros(k, dtype=bool)
        order = np.argsort(W, axis=None)[::-1]  # descending by weight
        matches = []
        for idx in order:
            i, j = np.unravel_index(idx, W.shape)
            w = W[i, j]
            if w <= 0.0:
                break  # remaining are <= 0
            if not used_q[i] and not used_k[j]:
                used_q[i] = True
                used_k[j] = True
                matches.append((i, j, float(w)))
        return matches


    def get_score_matching(
            self,
            query: List[str],
            keys: List[str] = None,
            tau: float = 0.4,
            return_tuple: bool = True,
            epsilon_max: float = 1e-3
        ):
        """
        Maximum-weight one-to-one matching scorer.

        Steps:
        1) \tilde S = max(0, (sim - tau)/(1 - tau)) in [0,1]
        2) greedy maximum-weight matching (each query/key used at most once)
        3) score by tuple (m, mean_weight, max_weight):
                - primary: coverage m (# matched pairs)
                - secondary: quality = mean weight over matched pairs
                - optional tie-break: max weight seen
        4) also return a scalar 'score_scalar' that respects the same ordering:
                score_scalar = m + mean_weight + epsilon_max * max_weight
            (lexicographic-compatible for sorting across entries)
        """
        sim = self._get_similarity_raw(query, keys)              # (q, k)
        Stilde = self._scale_and_threshold(sim, tau)             # (q, k) in [0,1]
        matches = self._greedy_max_weight_matching(Stilde)       # list of (i, j, w)

        m = len(matches)
        if m == 0:
            score_tuple = (0, 0.0, 0.0)
            score_scalar = 0.0
        else:
            weights = np.array([w for (_, _, w) in matches], dtype=float)
            mean_w = float(weights.mean())
            max_w = float(weights.max())
            score_tuple = (m, mean_w, max_w)
            # Scalar consistent with "rank by m, then mean, then max"
            score_scalar = m + mean_w + epsilon_max * max_w

        if return_tuple:
            return {"score_tuple": score_tuple,
                    "score_scalar": score_scalar,
                    "matches": matches}  # matches help with debugging/UX
        else:
            return float(score_scalar)




    def to_dict(self, include_embeddings: bool=True) -> dict:
        d = {
            "id":      self.id,
            "content": self.content,
            "keys":    list(self.keys),
        }
        if include_embeddings:
            d["embeddings"] = [np.array(emb) for emb in self.embeddings]
        return d

    def _from_dict(self, d: dict) -> None:
        self.content = d["content"]
        #self.add_keys(d["keys"])
        self.id = d.get("id", self.id)

        raw_keys = d.get("keys", [])
        if isinstance(raw_keys, (list, tuple, set, np.ndarray)):
            key_list = list(raw_keys)

        elif isinstance(raw_keys, str):
            try:
                decoded = json.loads(raw_keys)
                key_list = list(decoded) if isinstance(decoded, Iterable) else [raw_keys]
            except json.JSONDecodeError:
                key_list = [raw_keys]

        else:
            raise NotImplementedError("Wrong key type")

        self.add_keys(key_list)
        self.embeddings = [np.array(vec) for vec in d.get("embeddings", [])]
        return

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
        import html

        keys_html = "".join(f"<li>{html.escape(k)}</li>" for k in sorted(self.keys))
        content_html = html.escape(self.content).replace("\n", "<br>")

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
    <strong>Keys</strong>
    <ul>{keys_html}</ul>

    <strong>Content</strong>
    <p>{content_html}</p>
    </div>
    """



class Memory:
    memory : Dict[str, MemoryNode]
    keywords : Set[str]

    def __init__(self):
        self.memory : Dict[str, MemoryNode] = {}
        self.keywords : Dict[str, int] = {}
        self.score_matching = True
    
    def __size__(self) -> int:
        return len(self.memory.keys())
    
    def get_node(self, id):
        return self.memory.get(id)
    
    def delete_node(self, id):
        node : MemoryNode = self.get_node(id)
        if node is None:
            return None
        
        del self.memory[id]
        return node

    def add(self, node: MemoryNode):
        self.memory[node.id] = node
        for k in node.keys:
            self.update_keywords(k)
    
    def update_keywords(self, keyword):
        if keyword in self.keywords:
            self.keywords[keyword] += 1
        else:
            self.keywords[keyword] = 1
    
    def add_from_dict(self, node_dict: Dict) -> None:
        node = MemoryNode()
        node._from_dict(node_dict)
        self.add(node)

    def get_keywords(self, topk=50):
        '''Returns the topk most frequent keys in memory'''
        keys = self.keywords
        return sorted(keys, key=keys.get, reverse=True)[:topk]


    def get_nodes(self) -> List[MemoryNode]:
        nodes = []
        for node in self.memory.values():
            if len(node.keys) > 0:
                nodes.append(node)
                node.set_embeddings()
        return nodes
    
    def get_scores(self, nodes, queries: List[str], sensitivity: float = 0.01) -> np.ndarray:
        scores = []
        for node in nodes:
            if self.score_matching is True:
                s = node.get_score_matching(
                    query=queries,
                    tau=sensitivity,    
                    return_details=False       
                )
                scores.append(s)
            else:
                node_scores = node.get_score(queries, sensitivity=sensitivity)
                node_scores = np.array(node_scores)
                scores.append(node_scores)

        scores = np.array(scores) # m
        return scores
    
    """
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
    """

    def _recall(
        self,
        queries: List[str],
        max_recall: int = 5,
        sensitivity: float = 0.3,
        thres: float = 0.3,          # mean-weight threshold for matching mode
        min_matches: int = 1         # require at least this many matched pairs
    ) -> Dict[str, float]:
        """
        Returns {node_id: score_scalar}.
        - Matching mode: rank by tuple (m, mean_w, max_w) desc; filter by min_matches and mean_w >= thres.
        - Non-matching: rank by numeric score desc; filter by score > thres.
        """
        nodes = self.get_nodes()
        if not nodes:
            return {}

        excited_nodes: Dict[str, float] = {}

        if getattr(self, "score_matching", False):
            scored = []
            for node in nodes:
                res = node.get_score_matching(query=queries, tau=sensitivity, return_tuple=True)
                m, mean_w, max_w = res["score_tuple"]         # tuple used for ranking
                score_scalar = float(res["score_scalar"])
                scored.append((node, (m, mean_w, max_w), score_scalar))

            
            scored.sort(key=lambda x: x[1], reverse=True) # Lexicographic sort: highest coverage first, then mean weight, then max weight

            taken = 0
            for node, (m, mean_w, _), score_scalar in scored:
                if m < min_matches or mean_w < thres:
                    continue
                excited_nodes[node.id] = score_scalar
                taken += 1
                if taken >= max_recall:
                    break

            return excited_nodes

        scores = self.get_scores(nodes, queries, sensitivity)
        top_k_indices = np.argsort(-scores)

        for idx in top_k_indices[:max_recall]:
            s = float(scores[idx])
            if s <= thres:
                break
            node: MemoryNode = nodes[idx]
            excited_nodes[node.id] = s

        return excited_nodes

    def recall(self, queries: List[str], max_recall=5, sensitivity=0.3, thres=0.3) -> Dict[str, str]:
        excited_nodes = self._recall(queries, max_recall=max_recall, sensitivity=sensitivity, thres=thres)
        out = {}
        for id, score in excited_nodes.items():
            node = self.get_node(id)
            out[id] = node.__str__() 
        return out

    def modify_keywords(self, old_keys: Set[str], new_keys: List[str]) -> None:
        """
        Modify the keywords in the memory by removing old keys and adding new ones.
        """
        for key in old_keys:
            if key in self.keywords:
                self.keywords[key] -= 1
        for key in new_keys:
            self.update_keywords(key)
        
    
    def modify(self, id: str, new_stimuli: List[str] = None, new_content: str = None) -> None:
        node : MemoryNode = self.get_node(id)

        if node is None:
            return None, None
        
        if new_stimuli is not None:
            self.modify_keywords(node.keys, new_stimuli)
            node.remove_keys(node.keys, check=False)
            node.add_keys(new_stimuli)
            

        if new_content is not None:
            node.content = new_content
        
        return node, False


    def save_text(self, save_path):
        # save the memory to a .txt file
        memory_list = []
        for node in self.memory.values():
            memory_list.append(node.to_dict())
        # save by json
        with open(save_path, "w") as f:
            json.dump(memory_list, f)

    def save(self, save_path: str, include_embeddings=True) -> None:
        rows = [n.to_dict(include_embeddings=include_embeddings) for n in self.memory.values()]
        pd.DataFrame(rows).to_parquet(save_path, compression="zstd")

    def load_text(self, load_path):
        # load the memory from a file
        with open(load_path) as f:
            memory_list = json.load(f)
        for d in memory_list:
            try:
                node = MemoryNode().from_dict(d)
                self.memory[node.id] = node
            except ValueError as e:
                print(e, "for dictionary: ", d)
        self.load_keywords()

    def load(self, load_path: str, clear=False) -> None:
        df = pd.read_parquet(load_path)
        if clear is True:
            self.memory.clear()
        for _, row in df.iterrows():
            node = MemoryNode.from_dict(row.to_dict())
            self.memory[node.id] = node

        self.load_keywords()


    def load_keywords(self):
        """
        Load keywords from the memory nodes into the keyword dictionary.
        This is useful after loading a memory from a file.
        """
        self.keywords = {}
        for node in self.memory.values():
            for key in node.keys:
                self.update_keywords(key)
                

    def render_html(self):
        """
        Render an entire Memory object, laying each MemoryNode side-by-side.
        """
        import html
        node_snippets = [
            node.render_html()
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
        return combined_html
    
    def render(self):
        from IPython.display import HTML
        return HTML(self.render_html())
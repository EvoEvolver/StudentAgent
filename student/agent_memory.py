import json
import uuid
import numpy as np
from mllm import Chat, get_embeddings
from typing import Dict, List, Set

from student.bm25_indexing import get_bm25_score


class MemoryNode:
    id : str
    keys : Set[str]
    embeddings : List[str]
    content : str
    # assocations : list[str] # list of keys associated to this node
    

    def __init__(self, content: str = "", keys: List[str] = []):
        self.id = str(uuid.uuid4())
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
    

    def remove_keys(self, rem_keys: List[str]):
        assert isinstance(rem_keys, List)
        for key in rem_keys:
            assert isinstance(key, str)
            self.keys.remove(key)
        return self.keys


    def set_embeddings(self):
        self.embeddings = get_embeddings(list(self.keys))


    #def update_associations(self, new_associations):
    #    self.assocations = new_associations


    def get_score(self, input_keys, sensitivity=0, w: int =1):
        input_keys_embeddings = np.array(get_embeddings(input_keys))
        embeddings = np.array(self.embeddings)

        embed_similarity = np.dot(embeddings, input_keys_embeddings.T)
        embed_similarity = embed_similarity * (embed_similarity > sensitivity)
        embed_similarity = np.sum(embed_similarity, axis=0)
        
        bm25__similarity = get_bm25_score(self.keys, input_keys)
        
        return embed_similarity + w * bm25__similarity


    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "keys": self.keys,
        }

    def _from_dict(self, d):
        self.content = d["content"]
        self.keys = d["keys"]
        if "id" in d.keys():
            self.id = d["id"]
    
    def from_dict(cls, d):
        new_node = MemoryNode()  
        new_node._from_dict(d)
        return new_node

    def __str__(self):
        return f"{self.keys} : {self.content}"


class Memory:
    def __init__(self):
        self.memory : Dict[str, MemoryNode] = {}
    
    def __size__(self) -> int:
        return len(self.memory.keys())
    
    def add(self, node: MemoryNode):
        self.memory[node.id] = (node)
    
    def add_from_dict(self, node_dict: Dict) -> None:
        node = MemoryNode()
        node._from_dict(node_dict)
        self.add(node)


    def get_nodes(self) -> List[MemoryNode]:
        nodes = []
        for node in self.memory.values():
            if len(node.content) > 0:
                nodes.append(node)
                node.set_embeddings()
        return nodes


    def recall(self, queries: List[str], max_recall=5, sensitivity=0.01) -> Dict[str, str]:
        '''
        Performs a similarity search on the keys of the memory nodes (O(nodes^2)).

        Returns a dict[memory_node id -> content]
        '''
        scores = []
        nodes = self.get_nodes()
        if nodes is None:
            return []

        for node in nodes:
            scores.append(node.get_score(queries))

        scores = np.array(scores)
        score_summation_for_src = np.sum(scores, axis=0)
        score_norm_factor_for_src = score_summation_for_src + (score_summation_for_src == 0.0)
        scores = scores / score_norm_factor_for_src
        scores = np.sum(scores, axis=1)
        
        excited_nodes = {}
        if max_recall == 1:
            top_index = np.argmax(scores)
            node = nodes[top_index]
            excited_nodes[node.id] = node.content

        top_k_indices = np.argsort(-scores)
        
        
        for i in range(min(max_recall, len(top_k_indices))):
            if scores[top_k_indices[i]] <= sensitivity:
                break

            node : MemoryNode = nodes[top_k_indices[i]]
            excited_nodes[node.id] = node.content
        
        return excited_nodes
        
    
    
    def search(self, queries: List[str], top_k=10) -> Dict[str, str]:
        '''
        scores = []
        nodes = self.get_nodes()

        for node in nodes:
            scores.append(node.get_score(queries))
        
        scores = np.array(scores)
        score_summation_for_src = np.sum(scores, axis=0)
        # replace 0 entries by 1
        score_norm_factor_for_src = score_summation_for_src + (score_summation_for_src == 0.0)
        # normalize scores by the summation of each src
        scores = scores / score_norm_factor_for_src
        scores = np.sum(scores, axis=1)
        top_k_indices = np.argsort(scores)
        
        curr_index = len(top_k_indices) - 1
        top_excited_nodes = []
        while True:
            # break if the score is too low
            if scores[top_k_indices[curr_index]] <= 0.0001:
                break
            index_to_add = top_k_indices[curr_index]
            top_excited_nodes.append(nodes[index_to_add])
            if len(top_excited_nodes) == top_k:
                break
            curr_index -= 1
            if curr_index < 0:
                break
        return top_excited_nodes
        '''
        pass


    def search_and_filter(self, context, src_list: list[str], top_k=10, node_to_exclude=None):
        '''
        top_excited_nodes = self.search(src_list, top_k, node_to_exclude=node_to_exclude)
        prompts = [f"""
        You are required to select the most relevant memory that is related to the following context:
        <instruction>
        {context}
        </instruction>
        The following memories are retrieved from your knowledge base:
        """]
        for i, node in enumerate(top_excited_nodes):
            prompts.append(f"""
            <memory_{i}>
            {node.content()}
            </memory_{i}>
            """)
        prompts.append("""
        <output>
        You are required to output a JSON object with the following
        - analysis (str): the analysis of the memory that is most relevant to the instruction. mention their indices.
        - indices (list of int): the indices of the selected memories.
        </output>
        """)
        chat = Chat(dedent=True)
        chat += "\n".join(prompts)
        res = chat.complete(parse="dict", cache=True, expensive=True)
        filtered_nodes = []
        for i in res["indices"]:
            filtered_nodes.append(top_excited_nodes[i])
        return filtered_nodes
        '''
        pass


    def self_consistent_search(self, instruction: str, src_list: list[str], top_k=10):
        '''
        node_in_context = []
        src_list = src_list[:]
        while True:
            context = f"""
            You are search memory about `{instruction}`
            You have recalled the following memory:
            """
            for node in node_in_context:
                context += f"""
                <memory>
                {node.content()}
                </memory>
                """
                if not node.terminal:
                    src_list += node.src
            new_nodes = self.search_and_filter(context, src_list, top_k=top_k, node_to_exclude=node_in_context)
            node_in_context += new_nodes
            n_non_terminal_new_nodes = sum([1 for node in new_nodes if not node.terminal])
            if n_non_terminal_new_nodes == 0:
                break
        return node_in_context
        '''
        pass

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
            node = MemoryNode._from_dict(d)
            self.memory[node.id] = node


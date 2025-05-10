import json
import numpy as np
from mllm import Chat, get_embeddings

from student_draft.bm25_indexing import get_bm25_score


class MemoryNode:
    def __init__(self):
        self.content = ""
        self.src = []
        self.embedding = []
        self.data = None
        self.abstract = None
        self.terminal = True # whether the node can be used to recall more information

    def get_abstract(self):
        if self.abstract is None:
            return self.content
        return self.abstract

    def set_embedding(self):
        self.embedding = get_embeddings(self.src)

    def get_score(self, input_src_list):
        input_src_embeddings = get_embeddings(input_src_list)
        input_src_embeddings = np.array(input_src_embeddings)
        embedding = np.array(self.embedding)
        embed_similarity = np.dot(embedding, input_src_embeddings.T)
        # remove similarity below 0.6
        embed_similarity = embed_similarity * (embed_similarity > 0.2)
        # add up the similarity
        embed_similarity = np.sum(embed_similarity, axis=0)
        bm25__similarity = get_bm25_score(self.src, input_src_list)
        similarity = embed_similarity + bm25__similarity
        return similarity

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "content": self.content,
            "src": self.src,
            "data": self.data,
            "abstract": self.abstract,
            "terminal": self.terminal
        }

    def _from_dict(self, d):
        self.content = d["content"]
        self.src = d["src"]
        self.data = d["data"]
        self.abstract = d["abstract"]
        self.terminal = d["terminal"]

    @staticmethod
    def from_dict(d):
        if d["type"] == "MemoryNode":
            node = MemoryNode()
        elif d["type"] == "AssociateNode":
            node = AssociateNode()
        else:
            raise ValueError("Invalid type")
        node._from_dict(d)
        return node

class AssociateNode(MemoryNode):
    def __init__(self):
        super().__init__()
        self.associate: list[str] = []

    def to_dict(self):
        d = super().to_dict()
        d["associate"] = self.associate
        return d

    def _from_dict(self, d):
        self.content = d["content"]
        self.src = d["src"]


class Memory:
    def __init__(self):
        self.memory : list[MemoryNode] = []
    
    def __size__(self):
        return len(self.memory)

    def search(self, src_list: list[str], top_k=10, node_to_exclude=None):
        nodes = [node for node in self.memory if len(node.content) > 0]
        for node in nodes:
            node.set_embedding()
        scores = []
        for node in nodes:
            scores.append(node.get_score(src_list))
        scores = np.array(scores)
        score_summation_for_src = np.sum(scores, axis=0)
        # replace 0 entries by 1
        score_norm_factor_for_src = score_summation_for_src + (score_summation_for_src == 0.0)
        # normalize scores by the summation of each src
        scores = scores / score_norm_factor_for_src
        scores = np.sum(scores, axis=1)
        top_k_indices = np.argsort(scores)
        if node_to_exclude is None:
            node_to_exclude = []
        curr_index = len(top_k_indices) - 1
        top_excited_nodes = []
        while True:
            # break if the score is too low
            if scores[top_k_indices[curr_index]] <= 0.0001:
                break
            index_to_add = top_k_indices[curr_index]
            if nodes[index_to_add] not in node_to_exclude:
                top_excited_nodes.append(nodes[index_to_add])
            if len(top_excited_nodes) == top_k:
                break
            curr_index -= 1
            if curr_index < 0:
                break
        return top_excited_nodes


    def search_and_filter(self, context, src_list: list[str], top_k=10, node_to_exclude=None):
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
            {node.get_abstract()}
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


    def self_consistent_search(self, instruction: str, src_list: list[str], top_k=10):
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
                {node.get_abstract()}
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

    def save(self, save_path):
        # save the memory to a file
        memory_list = []
        for i, node in enumerate(self.memory):
            memory_list.append(node.to_dict())
        # save by json
        with open(save_path, "w") as f:
            json.dump(memory_list, f)

    def load(self, load_path):
        # load the memory from a file
        with open(load_path) as f:
            memory_list = json.load(f)
        for d in memory_list:
            node = MemoryNode.from_dict(d)
            self.memory.append(node)

def generate_non_stop_words(content):
    chat = Chat(dedent=True)
    chat += """
    <task>
    You are required to extract concepts from the given text.
    For example, concepts can be a character name, a technical term, a place name, etc.
    </task>
    """ + f"""
    You are required to generate concepts for the following text:
    <text>
    {content}
    </text>
    <output>
    You are required to output a JSON list of with a key
    "concepts" (list of string): a list of strings with each string being a concept extracted from the text
    </output>
    """
    res = chat.complete(cache=True, parse="dict")
    return res["concepts"]



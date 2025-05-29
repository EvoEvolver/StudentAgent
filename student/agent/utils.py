import json
import os
import requests
from typing import List

def pretty_json(json_dict):
    print(json.dumps(json_dict,indent=2, separators=(',', ': ')))


def get_prompt(file_name, path="./prompts/"):
    file_name = os.path.join(path, file_name)
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            content = f.read()
        return content
    return None


def instructions(s):
    return f"<instructions>{s}</instructions>"

def keyword(s):
    return f"<keyword name={s}/>"

def tool(s):
    return f"<tool name={s}/>"

def tool_response(tool_name, response, LIMIT=800):
    #return f"<tool response name={tool_name}>{response[:LIMIT]}</tool response>"
    return response[:LIMIT]

def recalled(s):
    return f"<recalled>{s}</recalled>"

def error(s):
    return f"<error>{s}</error>"

def file(name, content=""):
    return f"<file name={name}>{content}</file>"

def question(s):
    return f"<question>{s}</question>"

def context(s):
    return f"<context>{s}</context>"

def molecule(s):
    return f"<molecule name={s}/>"

def mol_name(name, wrong_names : List[str]):
    w_names = [f"<wrong_name name={w}/>" for w in wrong_names]
    return f"<use_name name={name}>{''.join(w_names)}</name>"


from rapidfuzz import process, fuzz
def quick_search(query, candidates, limit=10, score_cutoff=80):
    return process.extract(
        query,
        candidates,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff
    )


def request_by_post(url, payload=None):
    # Send a POST request
    response = requests.post(url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def get_file_hierarchy(path: str) -> dict:
    """
    Builds a nested dict representing the file/folder structure.

    Directories map to their own dict; files map to their full path (string).
    """
    tree = {}
    if not os.path.exists(path):
        return tree

    for entry in sorted(os.listdir(path)):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            tree[entry] = get_file_hierarchy(full_path)
        else:
            tree[entry] = full_path

    return tree


def all_files(path: str) -> List[str]:
    """
    Returns a list of all file paths under `path` (relative to `path`),
    plus any empty directories (with a trailing slash), also relative to `path`.
    """
    result: List[str] = []
    for root, dirs, files in os.walk(path):
        # files
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, start=path)
            result.append(rel)
        # empty dir?
        if not files and not dirs:
            reldir = os.path.relpath(root, start=path)
            # if root == path, os.path.relpath gives "."
            if reldir == ".":
                reldir = ""
            # ensure a single trailing slash
            result.append(reldir.rstrip(os.sep) + os.sep)
    return result


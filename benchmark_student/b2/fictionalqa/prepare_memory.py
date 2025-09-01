from student.agent import *
from student.agent.memory_rag import MemoryRAG, MemoryNodeRAG
import pandas as pd
import json
from student.agent.agent_baselines import BaselineAgent
import random

training_run_id = "002"

RAG_MEMORY_PATH = f"memory/combined_fqa_rag_{training_run_id}.parquet"
TRAINING_STUDENT_MEMORY_PATH = f"checkpoints/training_{training_run_id}"
SETUP_PATH = f"setup/setups_fqa_{training_run_id}.json"
STUDENT_MEMORY_PATH = f"checkpoints/memory_{training_run_id}"


AGENT_CONFIG = {
    "expensive": False,
    "provider": "anthropic",
    "cache": False
}



styles = [
    '_style_blog_num_000',
    '_style_blog_num_001',
    '_style_corporate_num_000',
    '_style_corporate_num_001',
    '_style_corporate_num_002',
    '_style_encyclopedia_num_000',
    '_style_encyclopedia_num_001',
    '_style_news_num_000',
    '_style_news_num_001',
    '_style_news_num_002',
    '_style_news_num_003',
    '_style_news_num_004',
    '_style_social_num_000',
    '_style_social_num_001',
    '_style_social_num_002'
]
styles_main = ["blog", "corporate", "encyclopedia", "news", "social", "fictsheet"][:1]
styles_nums = {"blog" : 2, "corporate" : 3, "encyclopedia" : 2, "news" : 5, "social" : 3, "fictsheet" : 1} # number of style variants

random.seed(68)
random.shuffle(styles_main)

data = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fict_qa/train-00000-of-00001.parquet")
fictsheets = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fictsheets/train-00000-of-00001.parquet")
fqa_mc = pd.read_parquet('https://huggingface.co/api/datasets/tomg-group-umd/fictionalqa_training_splits/parquet/fict_qa_obqa_blind_inf_ex_dedup_ds_Llama-3-2-3B-Instruct_scored_rowlimNone_altlimNone_topk4_seed1234_slim/train/0.parquet')
fictions = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fictions/train-00000-of-00001.parquet")


def data_from_question_id(question_id, data) -> pd.core.frame.DataFrame:
    entry = data[data['question_id'] == question_id]
    return entry[["event_id", "fiction_id", "question_id", "question_num", "fict", "question", "natural_answer"]]

# data_from_question_id("event_000_style_blog_num_000_question_003", data)

def load_fictsheets(event_id, fictsheets) -> str:
    # Returns the fictsheet as str
    fictsheets = fictsheets[["event_id", "fictsheet"]]
    return fictsheets[fictsheets['event_id'] == event_id]["fictsheet"].values[0]

# load_fictsheets("event_000", fictsheets)

def question_id_to_style(question_id) -> tuple[str, str]:
    # question_id: event_xxx_style_blog_num_xxx_question_xxx
    parts = question_id.split("_")
    style = parts[3]
    style_num = parts[5]
    return style, style_num

# question_id_to_style("event_000_style_blog_num_000_question_003")

def fqa_row_to_data(row, data, fictsheets) -> dict:
    question_id = row["question_id"]
    data_entry = data_from_question_id(question_id, data).to_dict()
    question_num = list(data_entry["question_num"].keys())[0]

    return {
        "event_id": row["event_id"],
        "fictsheet": load_fictsheets(row["event_id"], fictsheets),
        "question": data_entry["question"][question_num],
        "natural_answer": data_entry["natural_answer"][question_num],
        "question_id": data_entry["question_id"][question_num],
        "question_num": data_entry["question_num"][question_num],
        "fict": data_entry["fict"][question_num],
    }
# fqa_row_to_data(row, data, fictsheets)



def load_naive(all_ficts):
    memory_rag_naive = MemoryRAG()
    for d in all_ficts.values():
        new_node = MemoryNodeRAG(input=d)
        memory_rag_naive.add(new_node)

    #print({len(node.embeddings) for node in memory_rag_naive.memory.values()}) # all 0
    memory_rag_naive.get_nodes()
    #print({len(node.embeddings) for node in memory_rag_naive.memory.values()}) # all 1
    
    memory_rag_naive.save(RAG_MEMORY_PATH)

def load_student(all_ficts):
    teaching_prompt = "You are the world’s most studious detective of ficts, which are facts about fictitious stories that have never existed as facts about the real world. (NO verification required)"

    student = StudentAgent(**AGENT_CONFIG)
    student.reset_system_prompt(teaching_prompt, append=True)
    student.save(TRAINING_STUDENT_MEMORY_PATH)

    # Teaching procedure

    def train_memory(event, student):
        student.load(TRAINING_STUDENT_MEMORY_PATH)
        student.reset_chat()
        p = f"Ficts: {event}"
        student.run(p, remove_tools=["ask memory"])
        student.reset_chat()
        student.save(TRAINING_STUDENT_MEMORY_PATH)

    for j, context in enumerate(all_ficts.values()):
        print(j)
        train_memory(context, student)


    student.load(TRAINING_STUDENT_MEMORY_PATH)
    student.reset_conversation()
    student.save(STUDENT_MEMORY_PATH)
    
    tokens = student.sum_token_count()
    print(tokens["input_tokens"]/10**6 *4 + tokens["output_tokens"]/10**6 *0.8)


def main():
    n_event_ids = 1

    event_ids = fqa_mc["event_id"].unique()

    setups = {}
    num_questions = {style : 0 for style in styles_main}
    all_ficts = {}


    # for each event_id, one style OR fictsheet: 
    # training on the selected style, testing on selected questions per style

    for i, event_id in enumerate(event_ids[:n_event_ids]):
        fiction_ids = []
        questions = []

        # deterministically iterate over styles
        style_index = i%len(styles_main)
        style = styles_main[style_index]

        if style == "fictsheet":
            fictsheet = load_fictsheets(event_id, fictsheets)
            all_ficts[event_id] = fictsheet

            # collect all questions for each style
            for st in styles_main:
                if st == "fictsheet":
                    continue
                question_added = False
                
                for style_num in range(styles_nums[st]):
                    if question_added:
                        break
                    
                    fiction_id = f"{event_id}_style_{st}_num_00{style_num}"   
                    # n_questions[style_num] = len(fqa_mc[fqa_mc["fiction_id"] == fiction_id])
                    collected_questions = fqa_mc[fqa_mc["fiction_id"] == fiction_id]
                    
                    for q in collected_questions["question_id"].values:
                        if question_added:
                            break

                        questions.append(q)
                        fiction_ids.append(fiction_id)
                        question_added = True


                

        else:
            # style number such that maximum n_questions
            n_questions = {}
            for style_num in range(styles_nums[style]):
                fiction_id = f"{event_id}_style_{style}_num_00{style_num}"   
                n_questions[style_num] = len(fqa_mc[fqa_mc["fiction_id"] == fiction_id])

            max_style_num = max(n_questions, key=n_questions.get)
            
            fiction_id = f"{event_id}_style_{style}_num_00{max_style_num}"
            
            n_questions = len(fqa_mc[fqa_mc["fiction_id"] == fiction_id])

            # skip if no questions for any style number
            if n_questions == 0:
                print("No questions found for", fiction_id)
                continue
            fiction = fictions[fictions["fiction_id"] == fiction_id].iloc[0]["fiction"]
            
            all_ficts[event_id] = fiction
            questions = fqa_mc[fqa_mc["fiction_id"] == fiction_id]["question_id"].values
            for _ in range(len(questions)):
                fiction_ids.append(fiction_id)

        setups[event_id] = []

        for question_id, fiction_id in zip(questions, fiction_ids):
            mc_row = fqa_mc[fqa_mc["question_id"] == question_id].iloc[0].to_dict() 
            fict_row = fqa_row_to_data(data[data["question_id"] == question_id].iloc[0], data, fictsheets)

            setup = {
                "event_id": event_id,
                "context": fictsheet if style == "fictsheet" else fiction,
                
                "fiction_id": fiction_id,
                "question_id": question_id,
                "question": list(fict_row["question"]),
                "topk_choices" : list(mc_row["topk_choices"]), # list len(.) = 4
                "target" : mc_row["target"],
            }

            setups[event_id].append(setup)
            num_questions[style] += 1

    print("Number of questions in total per style: \t", num_questions)

    if not (n_event_ids == len(all_ficts.values())):
        raise ValueError("Mismatch in number of event_ids and fictions")

    # save setups
    with open(SETUP_PATH, "w") as f:
        json.dump(setups, f)


    load_naive(all_ficts)
    load_student(all_ficts)



if __name__ == "__main__":
    main()
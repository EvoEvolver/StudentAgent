import json
import os

from student.agent.agent_raspa import RaspaAgent

if __name__ == "__main__":

    knowledge = json.load(
        open(
            os.path.join(os.path.dirname(__file__), "..", "input", "knowledge.json"),
            "r",
        )
    )

    ag = RaspaAgent(path="../output/initial")

    for key in knowledge.keys():
        ag.get_memory().learn(knowledge[key])
    ag.get_memory()._save_to_file("../memory/initial_knowledge.json")
    print("Initial knowledge saved to ../memory/initial_knowledge.json")

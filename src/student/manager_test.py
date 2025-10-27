from dotenv import load_dotenv

from student.agent.agent_manager import ManagerAgent

load_dotenv()

agent = ManagerAgent(provider="openai")
agent.auto_run = True
reply = agent.run("setup monte carlo on methane in a box and execute it")
print(reply)

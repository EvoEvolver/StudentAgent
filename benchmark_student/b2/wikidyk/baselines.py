from student.agent import Agent

class BaselineAgent(Agent):
    def run_answerable(self, context, question):
        prompt = f"""You are tasked with a knowledge test. 
        You have to answer the <question> as precise and short as possible.

        Inputs:
        <question>
        {question}
        </question>
        <context>
        {context}
        </context>

        Instructions:
        - Output the precise answer for the <question>

        Example input: 
        <question>
        What is the capital of France?
        </question>
        <context>
        Paris is the capital of France.
        </context>

        Example outputs (IMPORTANT only precise + short answers. NO SENTENCES):
        Paris
        """
        return self.single_run(prompt, expensive=self.expensive, cache=self.cache, parse=None)
    
    def run_pretraining(self, question):
        prompt = f"""You are tasked with a knowledge test. 
        You have to answer the <question> as precise and short as possible.

        Inputs: <question>{question}</question>

        Instructions:
        - Output the precise answer for the <question>

        Example input: <question>What is the capital of France?</question>

        Example outputs (IMPORTANT only precise + short answers. NO SENTENCES): Paris
        """
        return self.single_run(prompt, expensive=self.expensive, cache=self.cache, parse=None)
    

    def run_evaluator(self, context, question, answer):
        prompt = f"""You are an evaluator for a knowledge test. 
        Your task is to decide if it is possible to give the correct <answer> the <question> only based on the <context>.
        Use only the information provided. Do not make assumptions beyond the given answers.

        Inputs:
        <question>
        {question}
        </question>

        <answer>
        {answer}
        </answer>

        <context>
        {context}
        </context>

        Instructions:
        - Output YES if <answer> can be correctly deduced from <context>
        - Output NO otherwise

        Example outputs (IMPORTANT never output anything else):
        YES
        NO
        """
        return self.single_run(prompt, expensive=self.expensive, cache=self.cache, parse=None)
    

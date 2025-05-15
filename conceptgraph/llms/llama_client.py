# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import timeit
from typing import Optional, Union
from collections import Counter

import torch
import transformers
from termcolor import colored

from conceptgraph.llms.base import AbstractLLMClient, AbstractPromptBuilder

default_model_id = "meta-llama/Meta-Llama-3.1-8B"

class LlamaClient(AbstractLLMClient):
    def chat_template(self, prompt):
        return f"\nUser: {prompt}Assistant:"

    def __init__(
        self,
        prompt: Optional[Union[str, AbstractPromptBuilder]],
        model_id: str = None,
        max_tokens: int = 1024,
        output_validation_function=None,
    ):
        super().__init__(prompt)
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_id is None:
            model_id = default_model_id

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = self.chat_template
        self.tokenizer.system_prompt = self.system_prompt
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up huggingface inference pipeline
        self.pipe = transformers.pipeline(
            "text-generation",
            # TODO: remove old code
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            # model=model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

        self.output_validation_function = output_validation_function

    def __call__(self, command: str, max_retries: int=30, verbose: bool = False):
        # return self.pipe(command, max_new_tokens=self.max_tokens)[0]["generated_text"].strip()
        if self.is_first_message():
            new_message = self.system_prompt + self.chat_template(command)
        else:
            new_message = self.chat_template(command)

        self.add_history(new_message)
        # Prepare the messages including the conversation history
        messages = self.get_history_as_str()

        assistant_response, time_taken = self._run_inference(messages)

        if self.output_validation_function is not None:
            is_valid, assistant_response = self.output_validation_function(assistant_response, command)
        else:
            is_valid, assistant_response = True, assistant_response

        retries = 0
        if not is_valid and retries < max_retries:
            assistant_response, more_time = self._run_inference(messages)
            time_taken += more_time
            is_valid = self._validate_output(assistant_response, command)
        
        if not is_valid:
            error = f"Invalid output: {assistant_response}"
            # raise ValueError(error)
            print(colored(error, 'yellow'))
            self.conversation_history.pop()
            return assistant_response

        # Add the assistant's response to the conversation history
        self.add_history({"role": "assistant", "content": " " + assistant_response})
        if verbose:
            print(f"Assistant response: {assistant_response}")
            print(f"Retries: {retries}")
            print(f"Total time taken: {time_taken:.2f}s")
            print(f"Average time taken: {time_taken / (retries + 1):.2f}s")
        return assistant_response
    
    # def format_llm_input(self, object_query: str, object_list: list):
    #     input_text = 'input list: [' + ', '.join(object_list) + '] input object: '+ object_query + '\n'
    #     return input_text
    
    def _run_inference(self, messages: Union[list, str]):
        t0 = timeit.default_timer()
        output = self.pipe(messages, max_new_tokens=self.max_tokens)
        t1 = timeit.default_timer()

        # Get response text
        assistant_response = output[0]["generated_text"].strip()
        assistant_response = assistant_response[len(messages) :].strip()

        # Hack: search for "User" in the response and remove everything after it
        user_idx = assistant_response.find("User")
        if user_idx != -1:
            assistant_response = assistant_response[:user_idx]
        assistant_idx = min(
            assistant_response.find("Assistant"), assistant_response.find("assistant")
        )
        if assistant_idx != -1:
            assistant_response = assistant_response[:assistant_idx]
        assistant_response = assistant_response.replace("\n", "")

        return assistant_response, t1 - t0
    
    def run_voting(self, command, max_retries: int=30, num_votes: int=3, verbose: bool=False,):
        # Vote on the best response
        reponses = []
        for _ in range(num_votes):
            response = self(command, max_retries, verbose)
            reponses.append(response)
            self.conversation_history.pop()
            query = self.conversation_history.pop()
        response = Counter(reponses).most_common(1)[0][0]
        confidence = Counter(reponses).most_common(1)[0][1] / num_votes
        # Add correct response to history
        self.add_history(query)
        self.add_history(' ' + response)
        return response, confidence
        

def validate_output_pocd(self, output: str, input_list: list):
    output_object = output.split('.')[0].strip().lower()
    if output_object in input_list:
        return True, output
    if output_object in ["static", "dynamic", "semi-static", "tatic", "ynamic", "emi-static"]:
        if output_object == "tatic":
            output = "static" + "." + output.split('.')[1]
        if output_object == "ynamic":
            output = "dynamic" + "." + output.split('.')[1]
        if output_object == "emi-static":
            output = "semi-static" + "." + output.split('.')[1]
        return True, output
    return False, output

def validate_output_similarity(self, output: str, input_list: list):
    output_list = output.split('.')[0].strip()
    try:
        # Remove square brackets and split by comma
        items = output_list.strip("[]").split(",")
        # Convert to float and check if all are positive
        numbers = [float(item.strip()) for item in items]
    except ValueError:
        print(f"Invalid output: {output_list} - not a list of numbers")
        return False, output
    
    all_positive = all(num > 0 for num in numbers)
    if not all_positive:
        print(f"Invalid output: {output_list} - not all positive")
        return False, output
        
    if len(output_list) != len(input_list):
        print(f"Invalid output: {output_list} - incorrect length")
        return False, output
    
    return True, numbers
    


if __name__ == "__main__":
    from conceptgraph.llms.prompts import HEATMAP_SYSTEM_PROMPT as heatmap_prompt
    from conceptgraph.llms.prompts import POCD_SYSTEM_PROMPT as pocd_prompt
    from conceptgraph.llms.prompts import OBJECT_SIMILARITY_SYSTEM_PROMPT
    from conceptgraph.utils.general_utils import ObjectClasses
    system_prompt = heatmap_prompt
    # system_prompt = pocd_prompt

    obj_classes = ObjectClasses(
        classes_file_path='/home/hornylemur/repos/concept-graphs/conceptgraph/scannet200_classes.txt',
        bg_classes=['wall', 'floor', 'ceiling'],
        skip_bg=False
    )

    system_prompt = OBJECT_SIMILARITY_SYSTEM_PROMPT % str(obj_classes.get_classes_arr()).replace("'", "")
    client = LlamaClient(system_prompt, max_tokens=512)

    def query_similarity_llm():
        input_str = 'chair'
        output = client(input_str, verbose=True)
        print(output)

    def query_heatmap_llm(object_query, object_list):
        prompt = 'input list: [' + ', '.join(object_list) + '] input object: '+ object_query + '\n'
        client(prompt, verbose=True)

    def test_heatmap():
        object_query = "spoon"
        object_list = ["table", "chair", "cabinet", "lamp"]
        query_heatmap_llm(object_query, object_list)
        
        object_query = "fork"
        object_list = ["oven", "chair", "couch", "lamp"]
        query_heatmap_llm(object_query, object_list)

        object_query = "lamp"
        object_list = ["bowl", "plate", "fork", "spoon", "chair", "piano", "bed", "couch"]
        query_heatmap_llm(object_query, object_list)

        object_query = "plate"
        object_list = ["bowl", "fork", "spoon", "chair", "table", "bed", "couch"]
        prompt = 'input list: [' + ', '.join(object_list) + '] input object: '+ object_query + '\n'
        r, c = client.run_voting(prompt, num_votes=5, verbose=False)
        print(colored(f"{r} with confidence {c}", 'red'))

    def query_pocd_llm(object_query):
        prompt = f'{object_query}\n'
        client(prompt, verbose=True)
    
    def test_pocd():
        object_query = "spoon"
        query_pocd_llm(object_query)

        object_query = "headphones"
        query_pocd_llm(object_query)

        object_query = "boulder"
        query_pocd_llm(object_query)

        object_query = "piano"
        prompt = f'{object_query}\n'
        r, c = client.run_voting(prompt, num_votes=5, verbose=False)
        print(colored(f"{r} with confidence {c}", 'red'))

    def test_pocd_random(trials: int=10):
        object_list = ["bowl", "plate", "fork", "spoon", "chair", "table", "bed", "couch", "lamp", \
                        "oven", "piano", "cabinet", "person", "headphones", "boulder", "rock", \
                        "tree", "flower", "plant", "pen", "pencil", "book", "paper", "computer", \
                        "monitor", "keyboard", "mouse", "desk", "window", "door", "wall", "floor"]
        length = len(object_list)
        import numpy as np
        for _ in range(trials):
            num = np.random.randint(0, length)
            object_query = object_list[num]
            # query_pocd_llm(object_query)
            prompt = f'{object_query}\n'
            r, c = client.run_voting(prompt, num_votes=5, verbose=False)
            print(colored(f"User: {object_query}", 'blue'))
            print(colored(f"Assistant: {object_query} is {r} with confidence {c}", 'red'))

    def test_heatmap_random(trials: int=10):
        object_list = ["bowl", "plate", "fork", "spoon", "chair", "table", "bed", "couch", "lamp", \
                        "oven", "piano", "cabinet", "person", "headphones", "boulder", "rock", \
                        "tree", "flower", "plant", "pen", "pencil", "book", "paper", "computer", \
                        "monitor", "keyboard", "mouse", "desk", "window", "door", "wall"]
        length = len(object_list)
        import numpy as np
        for _ in range(trials):
            num = np.random.randint(0, length)
            object_query = object_list[num]
            prompt = prompt = 'input list: [' + ', '.join(object_list) + '] input object: '+ object_query + '\n'
            r, c = client.run_voting(prompt, num_votes=5, verbose=False)
            print(colored(f"User: {object_query}", 'blue'))
            print(colored(f"Assistant: {object_query} is located near {r} Confidence {c}", 'red'))

    # test_heatmap()
    # test_pocd()
    # test_pocd_random(trials=5)
    # test_heatmap_random(trials=20)

    query_similarity_llm()

    print(client.get_history_as_str())

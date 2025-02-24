from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
from os import getenv
from conceptgraph.utils.prompts import POCD_SYSTEM_PROMPT, POCD_SYSTEM_PROMPT_SINGLE
from abc import ABC
from transformers import pipeline

class LLM(ABC):
    def __init__(self):
        super().__init__()

    def setup(self):
        raise NotImplementedError
    
    def format_llm_input(self):
        raise NotImplementedError
    
    def format_llm_output(self):
        raise NotImplementedError

    def run_inference(self):
        raise NotImplementedError
    
    def add_to_history(self):
        raise NotImplementedError
    
class POCDLLM(LLM):
    def __init__(self, model_id="google/gemma-2b-it", num_reprompt_tries=30, device="cuda"):
        super().__init__()
        supported_models = ["google/gemma-2b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
        if model_id == supported_models[0]:
            kwargs = {"torch_dtype": torch.float32}
        elif model_id == supported_models[1]:
            kwargs = {"torch_dtype": torch.float16}
        else:
            msg = f"{model_id} is not a support model. Supported models are {supported_models}."
            raise ValueError(msg)
        
        self.pipe = pipeline("text-generation", model=model_id, model_kwargs=kwargs, device="cuda")
        self.pocd_prompt = POCD_SYSTEM_PROMPT
        # self.pocd_single_prompt = POCD_SYSTEM_PROMPT_SINGLE

        self.last_input_in_string_form = None
        self.last_output_in_string_form = None
        self.prev_inputs_added_to_prompt = []

        self.num_reprompting_tries = num_reprompt_tries

    def _format_llm_input(self, object_list: list):
        input_text = '[' + ', '.join(object_list) + ']'
        self.last_input_in_string_form = input_text
        return input_text
    
    def _format_llm_output(self, output, input_text):
        output = output[0]['generated_text']
        output = output.removeprefix(self._append_to_prompt(input_text))
        self.last_output_in_string_form = output
        output = output.strip()[1:-1]
        output = [int(i) for i in output if i.isdigit()]
        return output
    
    # def _format_llm_output_single(self, output, input_text):
    #     output = output[0]['generated_text']
    #     output = output.removeprefix(self._append_to_prompt(input_text))
    #     self.last_output_in_string_form = output
    #     output = int(output.strip()[0])
    #     print(output)
    #     return output
    
    def _add_inference_to_history(self, single=False):
        if self.last_input_in_string_form not in self.prev_inputs_added_to_prompt:
            # if single:
            #     self.pocd_single_prompt += self.last_input_in_string_form + self.last_output_in_string_form + '\n\ninput: '
            # else:
            self.pocd_prompt += self.last_input_in_string_form + self.last_output_in_string_form + '\n\ninput: '
            self.prev_inputs_added_to_prompt.append(self.last_input_in_string_form)

    def _append_to_prompt(self, input_text, single=False):
        # if single:
        #     return self.pocd_single_prompt + input_text
        return self.pocd_prompt + input_text
    
    def _run_inference(self, input_text: str, max_response_length: int=4096):
        output = self.pipe(self._append_to_prompt(input_text), max_new_tokens=max_response_length)
        return output

    def _validate_output_length(self, object_list, output):
        if len(object_list) != len(output):
            return False
        return True
    
    def _validate_output_elements(self, output):
        for i in output:
            if i not in [0, 1, 2]:
                return False
        return True
    
    def run_inference(self, object_list: list, max_response_length: int=4096, add_to_history: bool=True):
        input_text = self._format_llm_input(object_list)
        output = self._run_inference(input_text, max_response_length)
        output = self._format_llm_output(output, input_text)

        retries = 0
        valid_length = self._validate_output_length(object_list, output)
        valid_elements = self._validate_output_elements(output)
        while (not valid_elements or not valid_length) and retries < self.num_reprompting_tries:
            retries += 1
            reprompt_text = self.last_input_in_string_form + self.last_output_in_string_form + '\n\n'
            if not valid_length:
                reprompt_text += f"The output list you provided has {len(output)} elements, but the input list has {len(object_list)} elements. Try again and make sure the output has the same number of elements as the input."
            if not valid_elements:
                reprompt_text += f"The output list you provided has elements that are not 0, 1, or 2. Try again and make sure all elements in the output list are either 0, 1, or 2. Remember tht 0 means a dynamic object, 1 means a semi-static object, and 2 means a static object."
            reprompt_text += '\n\ninput: ' + self.last_input_in_string_form
            
            output = self._run_inference(reprompt_text, max_response_length)
            output = self._format_llm_output(output, reprompt_text)
            valid_length = self._validate_output_length(object_list, output)
            valid_elements = self._validate_output_elements(output)
        
        if not valid_elements or not valid_length:
            # print("Last input: ", self.last_input_in_string_form)
            # print("Last output: ", self.last_output_in_string_form)
            error = f"Could not get a valid output after {self.num_reprompting_tries} tries. Valid length = {valid_length}. Valid elements = {valid_elements}."
            # raise ValueError(error)
            print(error)
            output = [0] * len(object_list)
            add_to_history = False

        if add_to_history:
            self._add_inference_to_history()
        
        return output, retries
    
    # def run_inference_single(self, object_list: list, max_response_length: int=4096, add_to_history: bool=True):
    #     type_list = []
    #     for obj in object_list:
    #         input_text = obj
    #         self.last_input_in_string_form = input_text
    #         output = self._run_inference(input_text, max_response_length)
    #         output = self._format_llm_output_single(output, input_text)

    #         retries = 0
    #         valid_output = output in [0, 1, 2]
    #         while (not valid_output) and retries < self.num_reprompting_tries:
    #             retries += 1
    #             reprompt_text = self.last_input_in_string_form + self.last_output_in_string_form + '\n\n'
    #             reprompt_text += f"The output you provided did not begin with an integer of 0, 1, or 2. Remember tht 0 means a dynamic object, 1 means a semi-static object, and 2 means a static object. Please try again."
    #             reprompt_text += '\n\ninput: ' + self.last_input_in_string_form
                
    #             output = self._run_inference(reprompt_text, max_response_length)
    #             output = self._format_llm_output_single(output, reprompt_text)
    #             valid_output = output in [0, 1, 2]
            
    #         if not valid_output:
    #             # print("Last input: ", self.last_input_in_string_form)
    #             # print("Last output: ", self.last_output_in_string_form)
    #             error = f"Could not get a valid output after {self.num_reprompting_tries} tries. Last output was: {self.last_output_in_string_form}."
    #             # raise ValueError(error)
    #             print(error)
    #             output = 0
    #             add_to_history = False

    #         if add_to_history:
    #             self._add_inference_to_history(single=True)

    #         type_list.append(output)
        
    #     return type_list
    

if __name__=='__main__':
    # HF_TOKEN = getenv('HF_TOKEN')
    # login(HF_TOKEN)
    # test_gemma_hello()
    # test_gemma_stationarity()
    pass
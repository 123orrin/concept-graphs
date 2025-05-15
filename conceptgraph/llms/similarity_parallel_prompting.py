import time
import asyncio
from openai import AsyncOpenAI
from conceptgraph.llms.prompts import OBJECT_SIMILARITY_PROMPT_SYSTEM, OBJECT_SIMILARITY_PROMPT_USER, OBJECT_SIMILARITY_PROMPT_ASSISTANT, object_similarity_prompt, process_similarity_response
from conceptgraph.utils.general_utils import ObjectClasses


class OpenAIAsyncClient:

    def __init__(self, object_list):
        self.client = AsyncOpenAI()
        if len(object_list) == 0:
            raise ValueError("The object list is empty.")
        self.object_list = object_list

    async def semantic_similarity_request(self, object_1: str, object_2: str):
        if ';' in object_1 or ';' in object_2:
            raise ValueError("Input objects should not contain semicolons.")

        async def semantic_similarity():
            completion = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "developer", "content": OBJECT_SIMILARITY_PROMPT_SYSTEM},
                    {"role": "user", "content": OBJECT_SIMILARITY_PROMPT_USER},
                    {"role": "assistant", "content": OBJECT_SIMILARITY_PROMPT_ASSISTANT},
                    {"role": "user", "content": object_similarity_prompt(object_1, object_2)},
                ]
            )

            response = completion.choices[0].message.content
            similarity_score, reason = process_similarity_response(response)

            return similarity_score, reason

        return await semantic_similarity()

    async def _query_semantic_similarity_parallel(self, object_list, query_object: str, debug=False):
        query_tasks = [self.semantic_similarity_request(o, query_object) for o in object_list]
        results = await asyncio.gather(*query_tasks)

        results_scores = [result[0] for result in results]
        results_reasons = [result[1] for result in results]

        if debug:
            print(results_scores)

            # Get the largest five scores and their corresponding reasons
            top_five_indices = sorted(range(len(results_scores)), key=lambda i: results_scores[i], reverse=True)[:5]
            top_five_scores = [results_scores[i] for i in top_five_indices]
            top_five_reasons = [results_reasons[i] for i in top_five_indices]
            top_five_objects = [object_list[i] for i in top_five_indices]

            # Print the results
            for i, (score, reason, obj) in enumerate(zip(top_five_scores, top_five_reasons, top_five_objects)):
                print(f"Rank {i + 1}: Object = {obj} Score = {score}, Reason = {reason}")

        return results_scores, results_reasons

    def query_semantic_similarity(self, query_object: str, debug=False):
        if len(query_object) == 0:
            raise ValueError("The query object is empty.")

        result_scores, result_reasons = asyncio.run(self._query_semantic_similarity_parallel(self.object_list, query_object, debug=debug))

        # Replace None in result_scores with 0 and print a warning
        for i, (score, response) in enumerate(zip(result_scores, result_reasons)):
            if score is None:
                result_scores[i] = 0
                print(f"Score was None. Object: '{self.object_list[i]}'. Reason: '{response}'")
            result_scores[i] /= 100.0

        return result_scores


# Example usage
def main():
    obj_classes = ObjectClasses(
        classes_file_path="/root/ros2_ws/src/concept-graphs/conceptgraph/scannet200_classes.txt",
        bg_classes=['wall', 'floor', 'ceiling'],
        skip_bg=False
    )
    object_list = obj_classes.get_classes_arr()
    print(object_list)

    client = OpenAIAsyncClient(object_list)
    similarities = client.query_semantic_similarity("chair", True)
    print(similarities)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nActual elapsed time: {elapsed_time} seconds")
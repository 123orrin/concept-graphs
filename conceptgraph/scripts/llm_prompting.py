import random
import time
import numpy as np

from openai import OpenAI

from prompts import prompts, semantic_types, constraint_types, constraint_groups


client = OpenAI()

def semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=None, debug=False):
    assert semantic_type in semantic_types, f"Semantic type must be one of {semantic_types}."

    sys = prompts["system"]
    user = prompts[semantic_type][0]
    assistant = prompts[semantic_type][1]
    prompt = prompts[semantic_type][2]

    def semantic_safety():
        start_time = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": sys},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": prompt(ee_object, scene_object, constraint_type)}
            ]
        )
        elapsed_time = time.time() - start_time

        response = completion.choices[0].message.content
        if debug:
            print(response)
            print(f"Elapsed time: {elapsed_time} seconds")

        return is_safe(response), elapsed_time

    return semantic_safety


def is_safe(response):
    # if response starts with "Yes." then return True else return False
    response = response.lower()
    if response.startswith("yes."):
        return True
    elif response.startswith("no."):
        return False
    else:
        print("Invalid response. Please respond with Yes. or No.")
        return None


def majority_vote(expression, repetitions=3, max_retries=3, debug=False):
    assert repetitions % 2 == 1, "Repetitions must be an odd number."

    votes = []
    timings = []
    for i in range(repetitions):
        vote = None
        retries = 0
        while vote is None:
            vote, timing = expression()
            if vote is None:
                print("Incorrect response format. Retrying...")
                retries += 1
                if retries > max_retries:
                    print("Maximum number of retries exceeded. Prompt may be insufficient.")
                    exit()
        votes.append(vote)
        timings.append(timing)

    if debug:
        print(votes)

    return votes.count(True) > votes.count(False), timings


def required_semantic_safety_constraints(ee_objects, semantic_types, scene_objects, constraint_types, constraint_groups, repetitions, debug=False, save_path=None):
    semantic_safety = []
    all_timings = []
    for ee_object in ee_objects:
        if "_" in ee_object:
            ee_object = " ".join(ee_object.split("_"))
        for semantic_type in semantic_types:
            for scene_object in scene_objects:
                for constraint_type in constraint_types:
                    is_semantically_safe, timings = majority_vote(semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=semantic_type, debug=debug), repetitions)
                    all_timings += timings
                    print("Majority vote: ", is_semantically_safe)
                    if not semantic_type == "spatial_relationship":
                        if semantic_type == "behavioral":
                            if not is_semantically_safe:
                                semantic_safety.append((ee_object, semantic_type, scene_object))
                        break
                    else:
                        if not is_semantically_safe:
                            already_covered = False
                            for constraint_in_group in constraint_groups[constraint_type]:
                                unsafe_constraint = (ee_object, semantic_type, scene_object, constraint_in_group)
                                if unsafe_constraint in semantic_safety:
                                    already_covered = True
                                    break
                            if not already_covered:
                                semantic_safety.append((ee_object, semantic_type, scene_object, constraint_type))
                if semantic_type == "pose":
                    if not is_semantically_safe:
                        semantic_safety.append((ee_object, semantic_type))
                    break
    
    print()
    print("Required semantic safety constraints: \n", semantic_safety)

    # Get statistics on timings
    all_timings = np.array(all_timings)

    print()
    print("Timing statistics")
    print("Number of ee objects: ", len(ee_objects))
    print("Number of scene objects: ", len(scene_objects))
    print("Number of semantic types: ", len(semantic_types))
    print("Number of constraint types: ", len(constraint_types))
    print("Number of repetitions: ", repetitions)
    print("Number of requests: ", len(all_timings))
    print("Total time: ", np.sum(all_timings))
    print("Mean: ", np.mean(all_timings))
    print("Median: ", np.median(all_timings))
    print("Standard deviation: ", np.std(all_timings))
    print("Minimum: ", np.min(all_timings))
    print("Maximum: ", np.max(all_timings))

    # Save timings to a file
    if save_path:
        print(f"Saving timings to {save_path}")
        np.save(save_path, all_timings)

    return semantic_safety


# For debugging purposes
def true_or_false():
    def random_boolean():
        # generate 0 or 1 randomly
        return bool(random.getrandbits(1))
    return random_boolean


if __name__ == "__main__":
    quick_test = False

    if quick_test:
        # semantic_type = "behavioral"
        semantic_type = "spatial_relationship"
           
        # ee_object = "cup of water"
        # ee_object = "bowl of soup"
        # ee_object = "cup of tea"
        # ee_object = "metal bowl"
        # ee_object = "ceramic bowl"
        ee_object = "plastic bowl"
        # ee_object = "bowl"
        # scene_object = "laptop"
        scene_object = "microwave"
        # constraint_type = "above"
        constraint_type = "inside"

        print(f"Semantic type: {semantic_type}")
        # is_semantically_safe = majority_vote(true_or_false(), repetitions=11)
        is_semantically_safe = majority_vote(semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=semantic_type, debug=True))
        print("Majority vote: ", is_semantically_safe)
    elif test_cup:
        # semantic_type = "spatial_relationship"
        # ee_object = "spray can"
        # scene_object = "stove"
        # constraint_type = "on"

        semantic_type = "spatial_relationship"
        ee_object = "cup of water"
        scene_object = "frying pan"
        constraint_type = "inside"

        print(f"Semantic type: {semantic_type}")
        is_semantically_safe = majority_vote(semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=semantic_type, debug=True))
        print("Majority vote: ", is_semantically_safe)

    else:
        # semantic_types = ["spatial_relationship", "behavioral", "pose"]
        constraint_types = ["above", "below", "around"]
        # constraint_types = ["above", "below", "around", "inside"]

        # ee_objects = ["cup of water", "dry sponge"]
        ee_objects = ["bottle of water"]
        # scene_objects = ["laptop", "books", "paper towel"]
        scene_objects = ["eggs"]
        repetitions = 3

        semantic_safety = required_semantic_safety_constraints(ee_objects, semantic_types, scene_objects, constraint_types, constraint_groups, repetitions=repetitions, debug=True)

        # for ee_object in ee_objects:
        #     print(f"End effector object: {ee_object}")
        #     for semantic_type in semantic_types:
        #         print(f"Semantic type: {semantic_type}")
        #         for scene_object in scene_objects:
        #             print(f"Scene object: {scene_object}")
        #             for constraint_type in constraint_types:
        #                 print(f"Constraint type: {constraint_type}")
        #                 is_semantically_safe = majority_vote(semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=semantic_type, debug=True))
        #                 print("Majority vote: ", is_semantically_safe)
        #                 if not semantic_type == "spatial_relationship":
        #                     break
        #             if semantic_type == "pose":
        #                 break

import random

from openai import OpenAI

from prompts import prompts, semantic_types, constraint_types


client = OpenAI()

def semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=None, debug=False):
    assert semantic_type in semantic_types, f"Semantic type must be one of {semantic_types}."

    sys = prompts["system"]
    user = prompts[semantic_type][0]
    assistant = prompts[semantic_type][1]
    prompt = prompts[semantic_type][2]

    def semantic_safety():
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": sys},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": prompt(ee_object, scene_object, constraint_type)}
            ]
        )

        response = completion.choices[0].message.content
        if debug:
            print(response)

        return is_safe(response)

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
    for i in range(repetitions):
        vote = None
        retries = 0
        while vote is None:
            vote = expression()
            if vote is None:
                print("Incorrect response format. Retrying...")
                retries += 1
                if retries > max_retries:
                    print("Maximum number of retries exceeded. Prompt may be insufficient.")
                    exit()
        votes.append(vote)

    if debug:
        print(votes)

    return votes.count(True) > votes.count(False)


def required_semantic_safety_constraints(ee_objects, semantic_types, scene_objects, constraint_types, repetitions, debug=False):
    semantic_safety = []
    for ee_object in ee_objects:
        for semantic_type in semantic_types:
            for scene_object in scene_objects:
                for constraint_type in constraint_types:
                    is_semantically_safe = majority_vote(semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=semantic_type, debug=debug), repetitions)
                    print("Majority vote: ", is_semantically_safe)
                    if not semantic_type == "spatial_relationship":
                        if semantic_type == "behavioral":
                            if not is_semantically_safe:
                                semantic_safety.append((ee_object, semantic_type, scene_object))
                        break
                    else:
                        if not is_semantically_safe:
                            semantic_safety.append((ee_object, semantic_type, scene_object, constraint_type))
                if semantic_type == "pose":
                    if not is_semantically_safe:
                        semantic_safety.append((ee_object, semantic_type))
                    break

    print("Required semantic safety constraints: ", semantic_safety)

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

    else:
        semantic_types = ["spatial_relationship", "behavioral", "pose"]
        # constraint_types = ["above", "below", "around"]
        # constraint_types = ["above", "below", "around", "inside"]

        ee_objects = ["cup of water", "dry sponge"]
        # scene_objects = ["laptop", "books", "paper towel"]
        scene_objects = ["laptop", "books"]

        semantic_safety = required_semantic_safety_constraints(ee_objects, semantic_types, scene_objects, constraint_types, repetitions=1, debug=True)

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

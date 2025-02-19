import random
import time
import numpy as np

import asyncio
from openai import AsyncOpenAI

from prompts import prompts, semantic_types, constraint_types, constraint_groups


client = AsyncOpenAI()

async def semantic_safety_request(ee_object, scene_object, constraint_type, semantic_type=None, debug=False):
    assert semantic_type in semantic_types, f"Semantic type must be one of {semantic_types}."

    sys = prompts["system"]
    user = prompts[semantic_type][0]
    assistant = prompts[semantic_type][1]
    prompt = prompts[semantic_type][2]

    async def semantic_safety():
        start_time = time.time()
        completion = await client.chat.completions.create(
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
            # print(f"Elapsed time: {elapsed_time} seconds")

        return is_safe(response), elapsed_time

    return await semantic_safety()

def is_safe(response):
    response = response.lower()
    if response.startswith("yes."):
        return True
    elif response.startswith("no."):
        return False
    else:
        print("Invalid response. Please respond with Yes. or No.")
        return None

async def majority_vote(expression_func, args, repetitions=3, max_retries=3, debug=False):
    assert repetitions % 2 == 1, "Repetitions must be an odd number."

    async def get_vote():
        vote = None
        retries = 0
        while vote is None and retries <= max_retries:
            vote, timing = await expression_func(*args, debug=debug)
            if vote is None:
                print("Incorrect response format. Retrying...")
                retries += 1
        if retries > max_retries:
            print("Maximum number of retries exceeded. Prompt may be insufficient.")
            return None, 0
        return vote, timing

    # Create tasks for all repetitions
    tasks = [get_vote() for _ in range(repetitions)]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    valid_results = [(vote, timing) for vote, timing in results if vote is not None]
    if not valid_results:
        return False, [0]
    
    votes, timings = zip(*valid_results)
    
    if debug:
        print(votes)

    return votes.count(True) > votes.count(False), list(timings)

async def required_semantic_safety_constraints(ee_objects, semantic_types, scene_objects, constraint_types, constraint_groups, repetitions, debug=False):
    semantic_safety = []
    all_timings = []
    results_cache = {}
    
    async def process_pose(ee_object):
        if "_" in ee_object:
            ee_object = " ".join(ee_object.split("_"))
            
        args = (ee_object, None, None, "pose")
        cache_key = (ee_object, "pose")
        
        if cache_key not in results_cache:
            is_semantically_safe, timings = await majority_vote(
                semantic_safety_request, args, repetitions=repetitions, debug=debug
            )
            all_timings.extend(timings)
            results_cache[cache_key] = is_semantically_safe
            print(f"Majority vote for pose {ee_object}: {is_semantically_safe}")
            
            if not is_semantically_safe:
                semantic_safety.append((ee_object, "pose"))
        
        return results_cache[cache_key]

    async def process_behavioral(ee_object, scene_object):
        if "_" in ee_object:
            ee_object = " ".join(ee_object.split("_"))
            
        args = (ee_object, scene_object, None, "behavioral")
        cache_key = (ee_object, "behavioral", scene_object)
        
        if cache_key not in results_cache:
            is_semantically_safe, timings = await majority_vote(
                semantic_safety_request, args, repetitions=repetitions, debug=debug
            )
            all_timings.extend(timings)
            results_cache[cache_key] = is_semantically_safe
            print(f"Majority vote for behavioral {ee_object}, {scene_object}: {is_semantically_safe}")
            
            if not is_semantically_safe:
                semantic_safety.append((ee_object, "behavioral", scene_object))
        
        return results_cache[cache_key]

    async def process_spatial(ee_object, scene_object, constraint_type):
        if "_" in ee_object:
            ee_object = " ".join(ee_object.split("_"))
            
        args = (ee_object, scene_object, constraint_type, "spatial_relationship")
        cache_key = (ee_object, "spatial_relationship", scene_object, constraint_type)
        
        if cache_key not in results_cache:
            is_semantically_safe, timings = await majority_vote(
                semantic_safety_request, args, repetitions=repetitions, debug=debug
            )
            all_timings.extend(timings)
            results_cache[cache_key] = is_semantically_safe
            print(f"Majority vote for spatial {ee_object}, {scene_object}, {constraint_type}: {is_semantically_safe}")
            
            if not is_semantically_safe:
                already_covered = False
                for constraint_in_group in constraint_groups[constraint_type]:
                    unsafe_constraint = (ee_object, "spatial_relationship", scene_object, constraint_in_group)
                    if unsafe_constraint in semantic_safety:
                        already_covered = True
                        break
                if not already_covered:
                    semantic_safety.append((ee_object, "spatial_relationship", scene_object, constraint_type))
        
        return results_cache[cache_key]

    # Process pose checks
    pose_tasks = [process_pose(ee_object) for ee_object in ee_objects]
    await asyncio.gather(*pose_tasks)
    
    # Process behavioral checks
    behavioral_tasks = [
        process_behavioral(ee_object, scene_object)
        for ee_object in ee_objects
        for scene_object in scene_objects
    ]
    await asyncio.gather(*behavioral_tasks)
    
    # Process spatial relationship checks
    spatial_tasks = [
        process_spatial(ee_object, scene_object, constraint_type)
        for ee_object in ee_objects
        for scene_object in scene_objects
        for constraint_type in constraint_types
    ]
    await asyncio.gather(*spatial_tasks)
    
    print("\nRequired semantic safety constraints:\n", semantic_safety)

    # Get statistics on timings
    all_timings = np.array(all_timings)

    print("\nTiming statistics")
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

    return semantic_safety


# For debugging purposes
def true_or_false():
    def random_boolean():
        # generate 0 or 1 randomly
        return bool(random.getrandbits(1))
    return random_boolean

# Example usage
async def main():
    # Your parameters here
    results = await required_semantic_safety_constraints(
        ee_objects=["cup_of_water", "plastic_bowl"],
        semantic_types=semantic_types,
        scene_objects=["eggs", "microwave"],
        constraint_types=constraint_types,
        constraint_groups=constraint_groups,
        repetitions=3,
        debug=True
    )
    return results

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    elapsed_time = time.time() - start_time
    print(f"\nActual elapsed time: {elapsed_time} seconds")

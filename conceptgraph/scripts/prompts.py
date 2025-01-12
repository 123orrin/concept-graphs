# SYSTEM:
system_prompt = "You are an intelligent assistant for robot-human interaction."

# 1. Spatial relationship prompts
# USER: 
spatial_relationship_user_prompt = """You are an intelligent assistant being used to guide a robot's decisions by 
telling it if certain moves are safe while holding an object. For the workflow, I will give you an object and 
some context and you will give me a response that begins strictly with Yes or with No. The second sentence is 
your reasoning. Do not be too cautious, only mark a move as unsafe if it poses a high risk to cause very 
significant damage. Less major damage or small chance of damage should be considered safe. 

Example:
A robotic manipulator is holding a dish of snacks in its end effector. The robotic manipulator is on a table. Your
job is to tell me if the object called plate is safe to go above. Please respond with Yes. or No. as the first
sentence and then a second sentence with your reasoning.

In this case, you should output: Yes. There is no worry of significant damage or harm by navigating over a plate. 

Example:
A robotic manipulator is holding a hot tea in its end effector. The robotic manipulator is on a table. Your job is
to tell me if the object called phone_1 is safe to go above. Please respond with Yes. or No. as the first sentence
and then a second sentence with your reasoning. 

In this case, you should output: No. If the cup of hot tea spills on a phone it may break the phone. 

Example:
A robotic manipulator is holding a milk carton in its end effector. The robotic manipulator is on a table. Your job
is to tell me if the object called refrigerator is safe to go inside. Please respond with Yes. or No. as the first
sentence and then a second sentence with your reasoning.

In this case, you should output: Yes. A milk carton can be safely placed inside refrigerator and is a common place 
for it to be stored.

Example:
A robotic manipulator is holding a phone in its end effector. The robotic manipulator is on a table. Your job is to 
tell me if the object called chair is safe to go inside. Please respond with Yes. or No. as the first sentence and 
then a second sentence with your reasoning.

In this case, you should output: No. A chair cannot be opened as it does not have a door or opening and the phone 
or chair may be damaged if it is forced inside.

Example:
A robotic manipulator is holding a container in its end effector. The robotic manipulator is on a table. Your job 
is to tell me if the object called oven is safe to go inside. Please respond with Yes. or No. as the first sentence 
and then a second sentence with your reasoning.

In this case, you should output: No. A general container may not be placed inside an oven as it may melt or catch fire. Only oven-safe 
containers should be placed inside an oven. 

Done examples. 

A robotic manipulator is holding a screwdriver. The robotic manipulator is on a table. Your job is to tell me if
the object called alarm_clock is safe to go above. Please respond with Yes. or No. as the first sentence and then a
second sentence with your reasoning."""

# ASSISTANT:
spatial_relationship_assistant_prompt = """Yes. It is safe to bring the screwdriver over the alarm_clock because even 
if the screwdriver fell on the alarm or came into contact the damage would not be very significant or costly."""

# USER:
def spatial_relationship_prompt(ee_object, scene_object, constraint_type):
    spatial_relation = "inside" if constraint_type == "inside" else "around"
    prompt = f"""A robotic manipulator is holding a {ee_object} in its end effector. The robotic 
    manipulator is navigating atop a desk. Your job is to tell me if the object called {scene_object} is safe for the robot 
    manipulator to go {constraint_type}. Assume the {ee_object} is planned to go {spatial_relation} the object at a distance of 2 cm 
    and that the robot moves slowly, it may contact but if so will be soft and will stop immediately. 
    If there is any ambiguity in the object, that could cause significant damage, rather mark it as unsafe.
    Please respond with Yes. or No. as the first sentence and then a second sentence with your reasoning."""
    return prompt

spatial_relationship_prompts = [spatial_relationship_user_prompt,
                                spatial_relationship_assistant_prompt,
                                spatial_relationship_prompt]

# 2. Behavioral prompts (caution)
# USER:
behavioral_user_prompt = """You are an intelligent assistant being used to guide a robot's decisions by telling it if extra 
caution should be observed while moving an object that it is holding. Extra caution should only be observed for objects that be 
risky to accelerate and move quickly. For the workflow, I will give you an object and some context and you will give me a 
response that begins strictly with Yes. or No. The second sentence is your reasoning.

Example:
A robotic manipulator is holding a paint_can in its end effector. Your job is to tell me if the object called paint_can is safe 
to move quickly or accelerate quickly towards the object called table. Please respond with Yes. or No. as the first sentence and 
then a second sentence with your reasoning. In this case you should output: No. The paint can might not be closed and sealed, so 
paint may spill out and make a mess on the table if quickly accelerated. 

Example:
A robotic manipulator is holding a tv_remote in its end effector. Your job is to tell me if the object called tv_remote is safe 
to move quickly or accelerate quickly towards the object called sofa. Please respond with Yes. or No. as the first sentence and 
then a second sentence with your reasoning. In this case you should output: Yes. There is no danger in moving the tv remote 
quickly close to the sofa. 

Done examples. 

A robotic manipulator is holding a screwdriver in its end effector. Your job is to tell me if the object called screwdriver is 
safe to move quickly or accelerate quickly towards the object called toolbox. Please respond with Yes. or No. as the first 
sentence and then a second sentence with your reasoning."""

# ASSISTANT:
behavioral_assistant_prompt = """Yes. A screwdriver is typically stable and can be moved quickly without risk, especially towards
 a toolbox, which is its expected destination."""

# USER:
def behavioral_prompt(ee_object, scene_object, constraint_type):
    prompt = f"""A robotic manipulator is holding a {ee_object} in its end effector. Your job is to tell me if the object called {ee_object} 
    is safe to move quickly or accelerate quickly towards the object called {scene_object}. Please respond with Yes. or No. as the 
    first sentence and then a second sentence with your reasoning."""
    return prompt

behavioral_prompts = [behavioral_user_prompt,
                      behavioral_assistant_prompt,
                      behavioral_prompt]

# 3. Pose-based prompts
# USER:
pose_user_prompt = """You are an intelligent assistant being used to guide a robot's decisions by telling it if it is safe to 
rotate an object that it is holding. Assume that the rotation is collision-free, but do not assume that there is nothing in or
on the object. For the workflow, I will give you an object and some context and you will give me a response that begins strictly
with Yes. or No. The second sentence is your reasoning.

Example:
A robotic manipulator is holding a paint_can in its end effector. Your job is to tell me if the object called paint_can is safe to 
rotate. Please respond with Yes. or No. as the first sentence and then a second sentence with your reasoning. In this case you 
should output: No. The paint can might not be closed and sealed, so paint may spill out and make a mess if rotated.

Done examples.

A robotic manipulator is holding a screwdriver in its end effector. Your job is to tell me if the object called screwdriver is 
safe to rotate. Please respond with Yes or No. as the first sentence and then a second sentence with your reasoning."""

# ASSISTANT:
pose_assistant_prompt = """Yes. Rotating the screwdriver does not pose a danger to the screwdriver or its surroundings and 
screwdrivers do not contain anything that may spill or fall if rotated."""

# USER:
def pose_prompt(ee_object, scene_object, constraint_type):
    prompt = f"""A robotic manipulator is holding a {ee_object} in its end effector. Your job is to tell me if the object called
    {ee_object} is safe to rotate, assuming that the rotation is collision-free. Please respond with Yes. or No. as the first 
    sentence and then a second sentence with your reasoning."""
    return prompt

pose_prompts = [pose_user_prompt,
                pose_assistant_prompt,
                pose_prompt]

prompts = {"system": system_prompt,
           "spatial_relationship": spatial_relationship_prompts,
           "behavioral": behavioral_prompts,
           "pose": pose_prompts}

semantic_types = ["spatial_relationship", "behavioral", "pose"]

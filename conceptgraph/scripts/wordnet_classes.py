from nltk.corpus import wordnet as wn


def download_corpus():
    # Download wordnet and brown corpus if not already downloaded
    try:
        wn.synsets('word')
    except LookupError:
        import nltk
        nltk.download('wordnet')


def get_tangible_objects():
    """
    Extract all tangible objects (physical entities) from WordNet.
    """
    # Get the "physical_entity" synset
    physical_entity_synset = wn.synset('physical_entity.n.01')
    
    # Recursively find all hyponyms of the physical_entity synset
    hyponyms = physical_entity_synset.closure(lambda s: s.hyponyms())
    
    # Collect all unique lemma names from the synsets
    tangible_objects = set(lemma for synset in hyponyms for lemma in synset.lemmas())
    
    return tangible_objects


def get_nonliving_tangible_objects():
    """
    Extract all tangible objects excluding living entities.
    """
    physical_entity_synset = wn.synset('physical_entity.n.01')
    living_thing_synset = wn.synset('living_thing.n.01')
    
    # Collect all tangible objects
    tangible_objects = set(lemma for synset in physical_entity_synset.closure(lambda s: s.hyponyms())
                           for lemma in synset.lemmas())
    
    # Collect all living entities
    living_entities = set(lemma for synset in living_thing_synset.closure(lambda s: s.hyponyms())
                          for lemma in synset.lemmas())
    
    # Exclude living entities
    nonliving_objects = tangible_objects - living_entities
    
    return nonliving_objects


def get_tangible_objects_refined():
    """
    Extract all nonliving tangible objects from WordNet.
    """
    # Start from the 'object' synset (physical things)
    object_synset = wn.synset('object.n.01')
    
    # Get all hyponyms of 'object'
    tangible_hyponyms = object_synset.closure(lambda s: s.hyponyms())
    
    # Collect all lemma names, excluding abstract and vague terms
    tangible_objects = set()
    for synset in tangible_hyponyms:
        for lemma in synset.lemmas():
            # Ensure it's not an abstract entity or a vague term
            if "abstract" not in [h.name() for h in synset.hypernyms()]:
                tangible_objects.add(lemma)
    
    return tangible_objects


def rank_artifacts_by_wordnet_frequency(artifact_lemmas):
    """
    Rank artifact lemmas based on their frequency in the WordNet corpus.
    """
    # Create a list of lemmas with their frequency counts
    ranked_artifacts = sorted(
        ((lemma.name(), lemma.count()) for lemma in artifact_lemmas),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked_artifacts


def main(max_num_artifacts, displayed_artifacts=10):
    download_corpus()

    # Get tangible objects from WordNet
    # tangible_objects = get_tangible_objects()
    # tangible_objects = get_nonliving_tangible_objects()
    tangible_objects = get_tangible_objects_refined()

    print(f"Number of tangible objects: {len(tangible_objects)}")
    print(list(tangible_objects)[:displayed_artifacts])

    # Rank tangible objects by frequency
    common_artifacts = rank_artifacts_by_wordnet_frequency(tangible_objects)

    # Display the most common artifacts
    print("Top {} most common artifacts:".format(displayed_artifacts))
    for artifact, freq in common_artifacts[:displayed_artifacts]:
        print(f"{artifact}: {freq}")

    common_artificats = list(map(lambda x: x[0], common_artifacts))
    common_artificats = common_artificats[:max_num_artifacts]
    
    list_of_desired_artifacts = ['robot', 'microwave', 'oven', 'refrigerator', 'toaster', 'dishwasher', 'washer', 'dryer', 'sink', 'faucet', 'cabinet', 'drawer', 'counter', 'table', 'chair', 'sofa', 'bed', 'lamp', 'curtain', 'blinds', 'rug', 'pillow', 'picture', 'television', 'remote', 'book', 'magazine', 'newspaper', 'plant', 'vase', 'flower', 'bowl', 'plate', 'cup', 'glass', 'fork', 'microwave door']

    for artifact in list_of_desired_artifacts:
        if artifact not in common_artificats:
            print(f"{artifact} not in common_artifacts")
        else:
            print(f"{artifact} in common_artifacts")

if __name__ == "__main__":
    max_num_artifacts = 5000
    displayed_artifacts = 30
    main(max_num_artifacts, displayed_artifacts=displayed_artifacts)

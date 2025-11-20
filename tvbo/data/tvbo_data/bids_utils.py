def get_unique_entity_values(bids_layout, key) -> set:
    """
    Get a set of all unique values for a given entity key from the BIDSLayout files.

    Args:
        bids_layout (BIDSLayout): The BIDSLayout object to extract entities from.
        key (str): The entity key to extract values for (e.g., 'atlas', 'space').

    Returns:
        set: A set of unique values for the specified entity key.
    """
    unique_values = set()
    files = bids_layout.get(return_type="file")

    for file in files:
        entities = bids_layout.parse_file_entities(file)
        if key in entities:
            unique_values.add(entities[key])

    return unique_values

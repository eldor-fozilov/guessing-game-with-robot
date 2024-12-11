from flask import current_app
from queue import Queue

def process_detected_objects(exclude_objects, detected_objects):

    flattened_objects = [obj for sublist in detected_objects for obj in sublist]

    if isinstance(exclude_objects, list) and exclude_objects:
        filtered_objects = [
            obj for obj in flattened_objects if obj["name"].lower() not in [ex.lower() for ex in exclude_objects]
        ]
    else:
        filtered_objects = flattened_objects

    unique_objects = list(set(obj["name"] for obj in filtered_objects))
    return filtered_objects, unique_objects



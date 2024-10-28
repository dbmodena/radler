import pandas as pd
import time
from utils import find_matching_neighbors, fusion, get_neighbors


def setup(task, ds, candidates, verbose):
    """
    Create a sketch for every record in the dataset
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param candidates: the list of candidate matching pairs of records
    :param verbose: show progress (Boolean)
    :return: the list of record sketches
    """

    setup_start_time = time.time()

    records = dict()

    neighbors, edge_weights = get_neighbors(ds, candidates)
    record_ids = list(neighbors.keys())

    for record_id in record_ids:
        sketch = dict()
        sketch["_id"] = record_id
        sketch["neighbors"] = neighbors[record_id]  # the ids of the candidate matching records
        sketch["matches"] = {record_id}  # the ids of the matching records
        sketch["solved"] = len(sketch["neighbors"]) == 1
        sketch["entity"] = fusion(ds, sketch["matches"], task.aggregations, task.time_attribute,
                                  task.default_aggregation) if sketch["solved"] else None
        records[record_id] = sketch

    if verbose:
        setup_time = time.time() - setup_start_time
        print("Setup completed for " + str(len(records)) + " records: " + str(setup_time) + " s.")

    return records


def cleaning(task, ds, gold, records, verbose):
    """
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param gold: the list of matches obtained using the selected matcher
    :param records: the list of record sketches
    :param verbose: show progress (Boolean)
    :return: the clean version of the entire dataset in the dataframe format
    """

    cleaning_start_time = time.time()

    entities = list()  # clean entities as dictionaries
    cleaned_ids = set()  # ids of the cleaned records

    num_comparisons = 0
    record_ids = list(records.keys())
    for record_id in record_ids:
        if record_id not in cleaned_ids:
            pivot_record = records[record_id]
            if not pivot_record["solved"]:
                comparisons = {pivot_record["_id"]: {pivot_record["_id"]}}  # track the performed comparisons
                pivot_record["matches"], comparisons, num_comparisons = find_matching_neighbors(pivot_record["_id"],
                                                                                                pivot_record["neighbors"],
                                                                                                pivot_record["matches"],
                                                                                                comparisons,
                                                                                                num_comparisons,
                                                                                                records,
                                                                                                gold)
                cleaned_ids = cleaned_ids.union(pivot_record["matches"])
                for record_id in pivot_record["matches"]:
                    del records[record_id]
                entity = fusion(ds, pivot_record["matches"], task.aggregations, task.time_attribute, task.default_aggregation)
                entity["matches"] = pivot_record["matches"]
                entities.append(entity)
                if verbose:
                    if len(entities) % 100 == 0:
                        print(str(len(entities)) + " entities cleaned (" \
                              + str(len(cleaned_ids)) + " out of " + str(len(record_ids)) + " records).")

    if verbose:
        cleaning_time = time.time() - cleaning_start_time
        print("Cleaning completed for " + str(len(entities)) + " entities.")
        print("Number of performed comparisons: " + str(num_comparisons) + ".")
        print("Elapsed time: " + str(cleaning_time) + " s.")

    return pd.DataFrame(entities)


def run(task, ds, gold, candidates, verbose=True):
    """
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param gold: the list of matches obtained using the selected matcher
    :param candidates: the list of candidate matching pairs of records
    :param verbose: show progress (Boolean)
    :return: the sample obtained after cleaning the entire dataset
    """

    # Create a sketch for every record
    records = setup(task, ds, candidates, verbose)

    # Clean all records in the dataset
    clean_ds = cleaning(task, ds, gold, records, verbose)

    return clean_ds

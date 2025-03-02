import math
import numpy as np
import pandas as pd
import polars as pl
import random
import statistics
import time
from utils import compute_distribution, to_sql


def get_neighbors(task, record_id):
    
    if task.mem_neighbors == "dictionary":
        return task.neighbors[record_id]


def update_neighbors(task, record_id, neighbors=None, drop=False):

    if task.mem_neighbors == "dictionary":
        task.neighbors[record_id] = None if drop else neighbors


def get_matches(task, record_id):

    if task.mem_matches == "dictionary":
        return task.matches[record_id]


def matching(task, left_id, right_id):
    """
    Check if the pair of records is present in the list of matches obtained using the selected matcher
    :param task: the object representing the task at hand
    :param left_id: the identifier of the left record
    :param right_id: the identifier of the right record
    :return: a Boolean value denoting if the pair of records is a match
    """

    return right_id in get_matches(task, left_id)


def find_matching_neighbors(task, record_id, neighbors, matches, comparisons, num_comparisons, sketches):
    """
    Find all matches of the current record (proceed recursively by following the matches)
    :param task: the object representing the task at hand
    :param record_id: the identifier of the current record
    :param neighbors: the set of neighbors of the current record
    :param matches: the set of matches of the current record
    :param comparisons: the dictionary to keep track of the performed comparisons
    :param num_comparisons: the number of performed comparisons
    :param sketches: the record sketches
    :return: the updated versions of matches, comparisons, and num_comparisons
    """

    for neighbor in list(neighbors):
        if neighbor not in matches and neighbor not in comparisons[record_id]:
            num_comparisons += 1
            comparisons[record_id].add(neighbor)
            if neighbor in comparisons.keys():
                comparisons[neighbor].add(record_id)
            else:
                comparisons[neighbor] = {neighbor, record_id}
            if matching(task, record_id, neighbor):
                matches.add(neighbor)
                matches, comparisons, num_comparisons = find_matching_neighbors(task, neighbor, get_neighbors(task, neighbor),
                                                                                matches, comparisons, num_comparisons, sketches)

    return matches, comparisons, num_comparisons


def fusion(task, ds, cluster):
    """
    Produce the clean entity from a cluster of matching records
    :param task: the object representing the task at hand
    :param ds: the dataset in the dataframe format
    :param cluster: the identifiers of the matching records
    :return: the clean entity as a dictionary
    """

    entity = dict()

    if isinstance(ds, pl.DataFrame):
        matching_records = ds.filter(pl.col("_id") == next(iter(cluster))) \
                           if len(cluster) == 1 else ds.filter(pl.col("_id").is_in(cluster))
        matching_records = matching_records.to_pandas()
    else:
        matching_records = ds.loc[ds["_id"].isin(cluster)]

    for attribute, aggregation in task.aggregations.items():
        if aggregation == "vote":
            modes = list(matching_records[attribute].mode())
            entity[attribute] = modes[0] if modes else np.nan
        elif aggregation == "min":
            entity[attribute] = matching_records[attribute].min()
        elif aggregation == "max":
            entity[attribute] = matching_records[attribute].max()
        elif aggregation == "avg":
            entity[attribute] = round(matching_records[attribute].mean(), 2)
        elif aggregation == "sum":
            entity[attribute] = round(matching_records[attribute].sum(), 2)
        elif aggregation == "random":
            entity[attribute] = np.random.choice(matching_records[attribute])
        elif aggregation == "concat":
            entity[attribute] = " ; ".join(matching_records[attribute])

    return entity


def compute_cost(task, neighbors):
    """
    Compute the cost component of the weighting scheme for the current record
    - Inversely proportional to the number of neighbors
    - Decimal value between 0 (min) and 1 (max)
    - Best case (max) when no comparison is needed
    :param task: the object representing the task at hand
    :param neighbors: the candidate matches (including the record itself)
    :return: the cost component of the weighting scheme for the current record
    """

    return (1 / len(neighbors)) if task.mode in {"complete", "cost"} else 1.0


def compute_benefits(task, neighbors, condition_records, valid_records):
    """
    Compute the probability of the record entity to belong to each group
    - Decimal value between 0 (min) and 1 (max)
    - Best case (max) when the entity belongs for sure to the group
    :param task: the object representing the task at hand
    :param neighbors: the candidate matches (including the record itself)
    :param condition_records: the records that satisfy each condition
    :param valid_records: the records with a valid value for each sample attribute
    :return: the benefit components of the weighting scheme for the current record
    """

    benefits = list()
    for group in task.groups:
        probs = list()  # probability of each condition
        set_to_none = False
        for i in range(task.num_attributes):
            attribute = task.sample_attributes[i]
            value = group[i]
            records = list(neighbors.intersection(condition_records[attribute][value]))
            num_records = len(records)
            if num_records > 0:
                num_neighbors = len(neighbors.intersection(valid_records[attribute]))
                probs.append((num_records / num_neighbors) if task.mode in {"complete", "benefit"} else 1.0)
            else:
                set_to_none = True
                break
        benefits.append(None if set_to_none else np.prod(probs))

    return benefits


def compute_weights(cost, benefits):
    """
    Compute the weight to assign to the current record for each group
    - Tradeoff between the cost and the benefit components of the weighting scheme
    - Decimal value between 0 (min) and 1 (max)
    :param cost: the cost component of the weighting scheme for the current record
    :param benefits: the benefit components of the weighting scheme for the current record
    :return: the weight to assign to the current record for each group
    """

    return [(cost * benefit) if benefit is not None else None for benefit in benefits]


def check_group(task, entity):
    """
    Check to which of the given groups (possibly none) the entity belongs
    :param task: the object representing the task at hand
    :param entity: the entity (as a dictionary)
    :return: the identifier of the group (None if it does not belong to any of the groups)
    """

    entity_values = tuple(entity[a] for a in task.sample_attributes)

    for i in range(0, task.num_groups):
        if entity_values == task.groups[i]:
            return i

    return None


def select_target_group(task, num_group_entities, active_groups):
    """
    Select the group to which the next entity should belong to minimize the divergence from the target distribution
    :param task: the object representing the task at hand
    :param num_group_entities: the number of entities from each group in the clean sample
    :param active_groups: the list of Boolean values stating if a group can still generate entities
    :return: the identifier of the target group to which the next entity should belong
    """

    divergence = list()

    for i in range(0, task.num_groups):
        new_distribution = compute_distribution([(num_group_entities[j] + 1) if j == i else num_group_entities[j]
                                                 for j in range(0, task.num_groups)])
        divergence.append(sum([abs(new_distribution[j] - task.target_distribution[j]) for j in range(0, task.num_groups)]))

    min_divergence = min(divergence)
    target_groups = [i for i in range(0, task.num_groups) if divergence[i] == min_divergence and active_groups[i]]

    return random.choice(target_groups) if len(target_groups) > 0 else None


def select_pivot_record_id(group_records, stochastic_acceptance_size, stochastic_acceptance_timeout, mode):

    pivot_id = None
    n = len(group_records)
    record_ids = list(group_records)

    if mode == "unweighed":
        return random.choice(record_ids)

    if n > stochastic_acceptance_size:
        loop_start_time = time.time()
        timeout = loop_start_time + stochastic_acceptance_timeout

        while time.time() < timeout:
            idx = random.randrange(n)
            weight = group_records[record_ids[idx]]

            if random.random() < weight:
                pivot_id = record_ids[idx]
                break

    if pivot_id is None:
        pivot_id = random.choices(record_ids, weights=list(group_records.values()), k=1)[0]

    return pivot_id


def setup(task, ds, run_stats, verbose):
    """
    Initialize the data structures
    :param task: the object representing the task at hand
    :param ds: the dataset in the dataframe format
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the initialized data structures
    - sketches: for each record, its sketch
    - group_records: for each group, the useful records with their weights
    - condition_records: for each condition, the records that satisfy it
    - valid_records: for each sample attribute, the records with a valid value
    """

    setup_start_time = time.time()

    record_ids = ds["_id"].to_list() if isinstance(ds, pl.DataFrame) else list(ds["_id"])

    sketches = [None] * len(record_ids)
    group_records = [dict() for _ in range(task.num_groups)]

    condition_records = {
        task.sample_attributes[i]: {
            v: set(ds.filter(pl.col(task.sample_attributes[i]) == v)["_id"].to_list()) if isinstance(ds, pl.DataFrame)
               else set(ds.query(to_sql(task.sample_attributes[i], v), engine="python")["_id"])
            for v in {g[i] for g in task.groups}
        } for i in range(task.num_attributes)
    }

    valid_records = {
        a: set(ds.select(["_id", a]).drop_nulls(subset=[a])["_id"].to_list()) if isinstance(ds, pl.DataFrame)
           else set(ds[["_id", a]].dropna(subset=[a])["_id"])
        for a in task.sample_attributes
    }

    num_sketches = 0
    for record_id in record_ids:
        sketch = dict()
        sketch["matches"] = {record_id}  # the represented cluster of matches
        neighbors = get_neighbors(task, record_id)  # load the candidate matches
        sketch["solved"] = len(neighbors) == 1  # no comparisons needed
        sketch["entity"] = None  # the entity obtained through data fusion
        sketch["group"] = None  # the group to which the entity belongs
        cost = compute_cost(task, neighbors)
        benefits = compute_benefits(task, neighbors, condition_records, valid_records)
        sketch["weights"] = compute_weights(cost, benefits)
        if sketch["solved"] and all(w is None for w in sketch["weights"]):
            continue
        sketches[record_id] = sketch
        for i in range(task.num_groups):
            if sketch["weights"][i] is not None:
                group_records[i][record_id] = sketch["weights"][i]
        num_sketches += 1

    setup_time = time.time() - setup_start_time

    if verbose:
        print("Setup completed for %s records (%s sketches): %s s." % (len(record_ids), num_sketches, setup_time))

    if run_stats is not None:
        run_stats.setup_time = setup_time

    return sketches, group_records, condition_records, valid_records, run_stats


def cleaning(task, ds, sketches, group_records, condition_records, valid_records, run_stats, verbose):
    """
    Produce the clean sample
    :param task: the object representing the task at hand
    :param ds: the dataset in the dataframe format
    :param sketches: the record sketches
    :param group_records: the useful records with their weights for each group
    :param condition_records: the records that satisfy each condition
    :param valid_records: the records with a valid value for each sample attribute
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the clean sample in the dataframe format and the updated metrics for the current run
    """

    cleaning_start_time = time.time()
    matching_time = 0.0
    fusion_time = 0.0

    entities = list()  # entities appearing in the clean sample (as dictionaries)
    fused_records = set()  # records fused to produce their clean entities

    active_groups = [True if len(group_records[i]) > 0 else False for i in range(task.num_groups)]
    num_group_entities = [0] * task.num_groups  # entities from each group in the clean sample

    iter_id = 0  # iteration counter
    num_comparisons = 0  # performed comparisons
    num_cleaned_entities = 0  # cleaned entities (including the ones not inserted into the clean sample)

    while len(entities) < task.sample_size:

        # Determine the target group to which the next entity should belong
        target_group = select_target_group(task, num_group_entities, active_groups)

        # Perform early stopping to avoid distorting the distribution of the clean sample
        if target_group is None:
            break

        # Select the pivot record
        group = group_records[target_group]
        pivot_id = select_pivot_record_id(group, task.stochastic_acceptance_size, task.stochastic_acceptance_timeout, task.mode)
        pivot_record = sketches[pivot_id]

        # Produce the clean entity (needed if the record has not been the pivot in a previous iteration)
        if pivot_record["entity"] is None:

            # Perform entity resolution if the pivot record has not been solved yet
            if not pivot_record["solved"]:
                neighbors = get_neighbors(task, pivot_id)
                comparisons = {pivot_id: {pivot_id}}  # track performed comparisons
                matching_start_time = time.time()
                pivot_record["matches"], comparisons, num_comparisons = find_matching_neighbors(task, pivot_id, neighbors,
                                                                                                pivot_record["matches"],
                                                                                                comparisons, num_comparisons,
                                                                                                sketches)
                matching_time += (time.time() - matching_start_time)
                compared_records = set(comparisons.keys()).difference(pivot_record["matches"])  # compared but did not match
                pivot_record["solved"] = True

                # Update weights for compared records
                for record_id in list(compared_records):
                    neighbors = get_neighbors(task, record_id).difference(pivot_record["matches"])
                    update_neighbors(task, record_id, neighbors)
                    old_weights = sketches[record_id]["weights"]
                    cost = compute_cost(task, neighbors)
                    benefits = compute_benefits(task, neighbors, condition_records, valid_records)
                    weights = compute_weights(cost, benefits)
                    sketches[record_id]["weights"] = weights
                    for i in range(task.num_groups):
                        if old_weights[i] is not None:
                            if weights[i] is not None:
                                group_records[i][record_id] = weights[i]
                            else:
                                del group_records[i][record_id]

            # Remove all matching records from the data structures
            for record_id in list(pivot_record["matches"]):
                weights = sketches[record_id]["weights"]
                for i in range(task.num_groups):
                    if weights[i] is not None:
                        del group_records[i][record_id]
                sketches[record_id] = None
                update_neighbors(task, record_id, drop=True)

            # Produce the clean entity and update the pivot record
            fused_records = fused_records.union(pivot_record["matches"])
            fusion_start_time = time.time()
            pivot_record["entity"] = fusion(task, ds, pivot_record["matches"])
            fusion_time += (time.time() - fusion_start_time)
            pivot_record["group"] = check_group(task, pivot_record["entity"])
            if pivot_record["group"] is not None:
                pivot_record["weights"] = [1.0 if i == pivot_record["group"] else None for i in range(task.num_groups)]
                group_records[pivot_record["group"]][pivot_id] = 1.0
                sketches[pivot_id] = pivot_record
            num_cleaned_entities += 1

        # Insert the entity to which the pivot record belongs into the clean sample
        if pivot_record["group"] == target_group:
            entity = pivot_record["entity"]
            entity["matches"] = pivot_record["matches"]
            entity["num_comparisons"] = num_comparisons
            entity["time"] = time.time() - cleaning_start_time
            entities.append(entity)
            num_group_entities[target_group] += 1
            del group_records[pivot_record["group"]][pivot_id]
            sketches[pivot_id] = None

        # Update the list of active groups
        for i in range(task.num_groups):
            if active_groups[i] and len(group_records[i]) == 0:
                active_groups[i] = False

        iter_id += 1

        if verbose:
            if iter_id % 100 == 0:
                print("%s iterations: %s entities cleaned (%s out of %s records)." %
                      (iter_id, len(entities), len(fused_records), len(sketches)))

    cleaning_time = time.time() - cleaning_start_time - matching_time - fusion_time

    if verbose:
        print("Cleaning completed for %s entities." % (len(entities)))
        print("Number of performed comparisons: %s." % (num_comparisons))
        print("Elapsed time: %s s." % (cleaning_time))

    if run_stats is not None:
        run_stats.cleaning_time = cleaning_time
        run_stats.num_entities = len(entities)
        run_stats.num_comparisons = num_comparisons
        run_stats.num_iterations = iter_id
        run_stats.num_cleaned_entities = num_cleaned_entities

    return pd.DataFrame(entities), run_stats


def run(task, ds, run_stats=None, verbose=True):
    """
    Perform deduplicated sampling on-demand
    :param task: the object representing the task at hand
    :param ds: the dataset in the dataframe format
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the obtained clean sample
    """

    if verbose:
        print("RadlER is running!")

    start_time = time.time()

    # Initialize the data structures
    sketches, group_records, condition_records, valid_records, run_stats = setup(task, ds, run_stats, verbose)

    # Produce the clean sample
    clean_sample, run_stats = cleaning(task, ds, sketches, group_records, condition_records, valid_records,
                                       run_stats, verbose)

    tot_time = time.time() - start_time

    if verbose:
        print("Total elapsed time: %s s." % (tot_time))

    if run_stats is not None:
        run_stats.tot_time = tot_time

        # Compute the number of entities per group in the sample
        num_group_entities = list()
        for g in task.groups:
            group_conditions = " and ".join([to_sql(task.sample_attributes[i], g[i]) for i in range(task.num_attributes)])
            group_subset = clean_sample.query(group_conditions, engine="python")
            num_group_entities.append(len(group_subset))
        run_stats.num_group_entities = num_group_entities
        run_stats.avg_cluster_size = statistics.mean([len(cluster) for cluster in clean_sample["matches"]])

        # Compute the progressive recall
        progressive_recall = list()
        num_steps = 20
        for step in range(0, num_steps + 1, 1):
            partial_comparisons = math.ceil((step / num_steps) * run_stats.num_comparisons)
            partial_entities = clean_sample[clean_sample["num_comparisons"] <= partial_comparisons]
            partial_recall = len(partial_entities)
            progressive_recall.append((partial_comparisons, partial_recall))
        run_stats.progressive_recall = progressive_recall

    return clean_sample, run_stats

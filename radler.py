import math
import numpy as np
import pandas as pd
import random
import statistics
import time
from utils import compute_distribution, find_matching_neighbors, fusion, get_neighbors


MAX_WEIGHT = 1.0
MIN_WEIGHT = 0.0


def compute_cost(neighbors, edge_weights, mode="edges"):
    """
    Compute the estimated cost of cleaning the entity to which the record belongs
    - Decimal value between 0 (min) and 1 (max)
    - Best case (max) when no comparison is needed
    :param neighbors: the candidate matches of the record (including itself)
    :param edge_weights: the weight of each neighbor
    :param mode: the function to use for computing the cost
    :return: the estimated cost of cleaning the current record
    """

    if mode == "edges":
        cost = 1 / len(neighbors)

    return cost


def compute_benefits(neighbors, edge_weights, groups, condition_records, valid_records, mode="weighed_edges"):
    """
    Compute the probability of the entity to which the record to belong to each given group
    - Decimal value between 0 (min) and 1 (max)
    - Best case (max) when the entity is totally sure to belong to the group
    :param neighbors: the candidate matches of a record (included itself)
    :param edge_weights: the weight of each neighbor
    :param group: the lists of conditions defining each group
    :param condition_records: the records that satisfy each condition
    :param mode: the function to use for computing the benefit
    :return: the list of estimated benefits of cleaning the current record for each group
    """

    benefits = list()

    # num_neighbors = len(neighbors)

    for group in groups:
        if mode == "edges" or mode == "weighed_edges":
            probs = list()  # probability of each condition
            set_to_none = False
            for condition in group:
                attribute = condition.split()[0]
                records = list(neighbors.intersection(condition_records[condition]))
                num_records = len(records)
                if num_records > 0:
                    num_neighbors = len(neighbors.intersection(valid_records[attribute]))
                    probs.append((num_records / num_neighbors) if mode == "edges"
                                 else (sum([edge_weights[record] for record in records]) / num_neighbors))
                else:
                    set_to_none = True
                    break
            benefits.append(None if set_to_none else np.prod(probs))

    return benefits


def compute_weights(cost, benefits, mode="product"):
    """
    Compute the tradeoff between cost and benefit of cleaning the entity to which the record belongs
    - Integer between 0 (min) and 1 (max)
    :param cost: the estimated cost of cleaning the current record
    :param benefits: the estimated benefit of cleaning the current record for each given group
    :param mode: the function to use for computing the weight
    :return: the tradeoff between cost and benefit of cleaning the current record for each given group
    """

    weights = list()

    for benefit in benefits:
        weights.append((cost * benefit) if benefit is not None else None)

    return weights


def check_group(entity, sample_attributes, groups):
    """
    Check to which of the given groups (possibly none) the entity belongs
    :param entity: the entity (as a dictionary)
    :param sample_attributes: the attributes used to define the groups
    :param groups: the lists of conditions defining each group
    :return: the identifier of the group (None if it does not belong to any of the groups)
    """

    entity_values = tuple(entity[attribute] for attribute in sample_attributes)

    for i in range(0, len(groups)):
        if entity_values == groups[i]:
            return i

    return None


def select_target_group(task, num_group_entities, group_records, active_groups):
    """
    Select the group of the next entity so to minimize the divergence from the target distribution
    :param task: the object representing the entity resolution task at hand
    :param num_group_entities: the number of entities for each group
    :param group_records: the weighted records for each group
    :param active_groups: the list of Boolean values stating if a group can still generate entities
    :return: the identifier of the target group for the next iteration
    """

    divergence = list()

    for i in range(0, task.num_groups):
        if active_groups[i]:
            new_distribution = compute_distribution([num_group_entities[j] + 1
                                                     if j == i else num_group_entities[j]
                                                     for j in range(0, task.num_groups)])
            divergence.append(sum([abs(new_distribution[j] - task.target_distribution[j])
                                   for j in range(0, task.num_groups)]))
        else:
            divergence.append(math.inf)

    min_divergence = min(divergence)
    target_groups = {i for i in range(0, task.num_groups) if divergence[i] == min_divergence}

    return list(target_groups)[0] if len(target_groups) == 1 else random.choice(list(target_groups))


def setup(task, ds, candidates, run_stats, verbose):
    """
    Create a sketch and compute the initial weight per group for every record in the dataset
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param candidates: the list of candidate matching pairs of records
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the record sketches, the weighted records for each group, the records that satisfy each condition
    """

    setup_start_time = time.time()

    records = dict()
    group_records = [dict() for _ in range(0, task.num_groups)]
    condition_records = {condition: set(ds.query(condition, engine="python")["_id"])
                         for group in task.sql_groups for condition in group}
    valid_records = {s: set(ds[["_id", s]].dropna()["_id"]) for s in task.sample_attributes}

    neighbors, edge_weights = get_neighbors(ds, candidates)
    record_ids = list(neighbors.keys())

    for record_id in record_ids:
        sketch = dict()
        sketch["_id"] = record_id
        sketch["neighbors"] = neighbors[record_id]  # the ids of the candidate matching records
        sketch["edge_weights"] = edge_weights[record_id]  # the weight of the candidate matching records
        sketch["matches"] = {record_id}  # the ids of the matching records
        sketch["solved"] = len(sketch["neighbors"]) == 1
        if sketch["solved"]:
            sketch["entity"] = fusion(ds, sketch["matches"], task.aggregations, task.time_attribute, task.default_aggregation)
            sketch["group"] = check_group(sketch["entity"], task.sample_attributes, task.groups)
            sketch["weights"] = [MAX_WEIGHT if i == sketch["group"] else None
                                 for i in range(0, task.num_groups)]
        else:
            sketch["entity"] = None
            sketch["group"] = None
            cost = compute_cost(sketch["neighbors"], sketch["edge_weights"], task.cost_mode)
            benefits = compute_benefits(sketch["neighbors"], sketch["edge_weights"], task.sql_groups,
                                        condition_records, valid_records, task.cost_mode)
            sketch["weights"] = compute_weights(cost, benefits)
        records[record_id] = sketch
        for i in range(0, task.num_groups):
            if sketch["weights"][i] is not None:
                group_records[i][record_id] = sketch["weights"][i]

    setup_time = time.time() - setup_start_time

    if verbose:
        print("Setup completed for " + str(len(records)) + " records: " + str(setup_time) + " s.")

    if run_stats is not None:
        run_stats.setup_time = setup_time

    return records, group_records, condition_records, valid_records, run_stats


def cleaning(task, ds, gold, records, group_records, condition_records, valid_records, start_time, mode, run_stats, verbose):
    """
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param gold: the list of matches obtained using the selected matcher
    :param records: the list of record sketches
    :param group_records: the weighted records for each group
    :param condition_records: the records that satisfy each condition
    :param start_time: the start time of the RadlER algorithm
    :param mode: the operating mode ("random": weighted random selection, "cheapest": deterministic cheapest selection)
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the clean sample in the dataframe format
    """

    cleaning_start_time = time.time()

    entities = list()  # clean entities as dictionaries
    cleaned_ids = set()  # ids of the cleaned records

    active_groups = [True if len(group_records[i]) > 0 else False for i in range(0, task.num_groups)]
    num_group_entities = [0 for _ in range(0, task.num_groups)]
    num_comparisons = 0
    num_cleaned_entities = 0
    num_records = len(records)
    iter_id = 0

    while any(active_groups):

        if task.stop_mode == "size":
            if len(entities) == task.sample_size:
                break
        elif task.stop_mode == "time":
            if time.time() - start_time >= task.time_limit:
                break
        
        if task.early_stopping and not all(active_groups):
            break

        target_group = select_target_group(task, num_group_entities, group_records, active_groups)
        if mode == "random":
            record_id = random.choices(list(group_records[target_group].keys()),
                                       weights=list(group_records[target_group].values()), k=1)[0]
        elif mode == "cheapest":
            record_id = sorted(list(group_records[target_group].items()), key=lambda x: x[1], reverse=True)[0][0]

        pivot_record = records[record_id]

        # If the pivot record still needs to be solved, perform entity resolution
        clean = not pivot_record["solved"]
        if clean:
            comparisons = {pivot_record["_id"]: {pivot_record["_id"]}}  # track the performed comparisons
            pivot_record["matches"], comparisons, num_comparisons = find_matching_neighbors(pivot_record["_id"],
                                                                                            pivot_record["neighbors"],
                                                                                            pivot_record["matches"],
                                                                                            comparisons,
                                                                                            num_comparisons,
                                                                                            records,
                                                                                            gold)
            cleaned_ids = cleaned_ids.union(pivot_record["matches"])
            compared_ids = set(comparisons.keys()).difference(pivot_record["matches"])  # compared but not matched

            # Remove all matching records
            for record_id in list(pivot_record["matches"]):
                weights = records[record_id]["weights"]
                for i in range(0, task.num_groups):
                    if weights[i] is not None:
                        del group_records[i][record_id]
                del records[record_id]

            # Update weights for compared records
            for record_id in list(compared_ids):
                neighbors = records[record_id]["neighbors"].difference(pivot_record["matches"])
                records[record_id]["neighbors"] = neighbors
                old_weights = records[record_id]["weights"]
                cost = compute_cost(neighbors, records[record_id]["edge_weights"], task.cost_mode)
                benefits = compute_benefits(neighbors, records[record_id]["edge_weights"], task.sql_groups,
                                            condition_records, valid_records, task.benefit_mode)
                weights = compute_weights(cost, benefits, task.weight_mode)
                records[record_id]["weights"] = weights
                for i in range(0, task.num_groups):
                    if old_weights[i] is not None:
                        if weights[i] is not None:
                            group_records[i][record_id] = weights[i]
                        else:
                            del group_records[i][record_id]

            # Update the pivot record
            pivot_record["solved"] = True
            pivot_record["neighbors"] = {pivot_record["_id"]}
            pivot_record["entity"] = fusion(ds, pivot_record["matches"], task.aggregations,
                                            task.time_attribute, task.default_aggregation)
            pivot_record["group"] = check_group(pivot_record["entity"], task.sample_attributes, task.groups)
            pivot_record["weights"] = [MAX_WEIGHT if i == pivot_record["group"] else None
                                       for i in range(0, task.num_groups)]
            records[pivot_record["_id"]] = pivot_record
            for i in range(0, task.num_groups):
                if pivot_record["weights"][i] is not None:
                    group_records[i][pivot_record["_id"]] = pivot_record["weights"][i]
            num_cleaned_entities += 1

        # Insert the entity of the pivot record into the clean sample
        if pivot_record["group"] == target_group:
            entity = pivot_record["entity"]
            entity["matches"] = pivot_record["matches"]
            entity["num_comparisons"] = num_comparisons
            entity["time"] = time.time() - cleaning_start_time
            entities.append(entity)
            num_group_entities[target_group] += 1
            del records[pivot_record["_id"]]
            del group_records[pivot_record["group"]][pivot_record["_id"]]

        for i in range(0, task.num_groups):
            if active_groups[i] and len(group_records[i]) == 0:
                active_groups[i] = False

        if verbose:
            if iter_id % 100 == 0:
                print("Iteration " + str(iter_id) + ": " + str(len(entities)) + " entities cleaned (" \
                      + str(len(cleaned_ids)) + " out of " + str(num_records) + " records).")

        iter_id += 1

    cleaning_time = time.time() - cleaning_start_time

    if verbose:
        print("Cleaning completed for " + str(len(entities)) + " entities.")
        print("Number of performed comparisons: " + str(num_comparisons) + ".")
        print("Elapsed time: " + str(cleaning_time) + " s.")

    if run_stats is not None:
        run_stats.cleaning_time = cleaning_time
        run_stats.num_entities = len(entities)
        run_stats.num_comparisons = num_comparisons
        run_stats.num_cleaned_entities = num_cleaned_entities

    return pd.DataFrame(entities), run_stats


def run(task, ds, gold, candidates, mode="random", run_stats=None, verbose=True):
    """
    :param task: the object representing the entity resolution task at hand
    :param ds: the dataset in the dataframe format
    :param gold: the list of matches obtained using the selected matcher
    :param candidates: the list of candidate matching pairs of records
    :param mode: the operating mode ("random": weighted random selection, "cheapest": deterministic cheapest selection)
    :param run_stats: the object used to collect the metrics for the current run
    :param verbose: show progress (Boolean)
    :return: the obtained clean sample
    """

    start_time = time.time()

    # Create a sketch for every record
    records, group_records, condition_records, valid_records, run_stats = setup(task, ds, candidates, run_stats, verbose)

    # Clean all records in the dataset
    clean_sample, run_stats = cleaning(task, ds, gold, records, group_records, condition_records, valid_records,
                                       start_time, mode, run_stats, verbose)

    tot_time = time.time() - start_time

    if verbose:
        print("Total elapsed time: " + str(tot_time) + " s.")

    if run_stats is not None:
        run_stats.tot_time = tot_time

        # Compute the number of entities per group in the sample
        num_group_entities = list()
        avg_proxy_intra = list()
        for i in range(0, len(task.groups)):
            group_conditions = " and ".join(task.sql_groups[i])
            group_subset = clean_sample.query(group_conditions, engine="python")
            num_group_entities.append(len(group_subset))
            proxy_values = [x for x in list(group_subset[task.proxy_attribute]) if not math.isnan(x)]
            avg_proxy_intra.append(statistics.mean(proxy_values) if len(proxy_values) > 0 else None)
        run_stats.num_group_entities = num_group_entities
        run_stats.avg_proxy_intra = avg_proxy_intra

        run_stats.avg_cluster_size = statistics.mean([len(cluster) for cluster in clean_sample["matches"]])
        run_stats.avg_proxy_inter = statistics.mean([x for x in list(clean_sample[task.proxy_attribute]) if not math.isnan(x)])

        progressive_recall = list()
        progressive_error = list()
        num_steps = 20
        for step in range(0, num_steps + 1, 1):
            partial_comparisons = math.ceil((step / num_steps) * run_stats.num_comparisons)
            partial_entities = clean_sample[clean_sample["num_comparisons"] <= partial_comparisons]
            partial_recall = len(partial_entities)
            progressive_recall.append((partial_comparisons, partial_recall))
            group_error = list()
            for i in range(0, len(task.groups)):
                group_conditions = " and ".join(task.sql_groups[i])
                group_subset = partial_entities.query(group_conditions, engine="python")
                group_distribution = (len(group_subset) / partial_recall) if partial_recall > 0 else 0
                group_error.append(abs(task.target_distribution[i] - group_distribution))
            progressive_error.append((partial_comparisons, group_error, sum(group_error)))
        run_stats.progressive_recall = progressive_recall
        run_stats.progressive_error = progressive_error

    return clean_sample, run_stats

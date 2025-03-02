import itertools
import networkx as nx
import pandas as pd
import random


def main():
    ds = pd.read_csv("../dataset.csv")
    record_ids = list(ds["_id"])
    matches = pd.read_csv("../matches.csv")
    matches = [tuple(row) for row in matches.itertuples(index=False)]
    set_matches = set(matches)
    num_matches = len(matches)

    g = nx.Graph()
    g.add_nodes_from(record_ids)
    g.add_edges_from(matches)
    clusters = [c for c in nx.connected_components(g)]
    print("Nodes: %s" % (len(g.nodes)))
    print("Edges: %s" % (len(g.edges)))
    print("Clusters: %s" % (len(clusters)))

    groups = {'sony', 'canon', 'nikon', 'fujifilm', 'samsung', 'olympus', 'panasonic', 'kodak', 'dahua', 'casio'}
    cluster_ids = [i for i in range(len(clusters))]
    cluster_to_groups = {key: set() for key in range(len(clusters))}
    group_to_clusters = {key: set() for key in groups.union({'other'})}
    for i in range(len(clusters)):
        records = ds[ds["_id"].isin(clusters[i])]
        cluster_groups = set(records["brand"])
        cluster_groups = cluster_groups.intersection(groups).union({'other'} if len(cluster_groups.difference(groups)) > 0 else {})
        cluster_to_groups[i] = cluster_groups
        for g in list(cluster_groups):
            group_to_clusters[g].add(i)
    groups = groups.union({'other'})

    recall_levels = [1.0, 0.95, 0.9, 0.85, 0.8]
    recall_matches = list()
    for l in recall_levels:
        num_tp = int(l * num_matches)
        if l == 1.0:
            recall_matches.append(set(matches))
        else:
            recall_matches.append(set(random.sample(recall_matches[-1], num_tp)))

    candidate_sets = list()
    false_positives = set()
    queue = set()
    block_id = 0
    for n in [1, 2, 4, 8, 16]:
        for tp in recall_matches:
            num_fp = (n * num_matches) - len(tp)
            iter_id = 0
            if len(queue) > 0:
                queue_samples = set(random.sample(list(queue), min(len(queue), num_fp - len(false_positives))))
                false_positives = false_positives.union(queue_samples)
                queue = queue.difference(queue_samples)
            while len(false_positives) < num_fp:
                cl_id_1 = random.choice(cluster_ids)
                ratio = 3
                g_cl_id_2 = random.choice(list(cluster_to_groups[i] if (iter_id + 1) % (ratio + 1) != 0
                                               else groups.difference(cluster_to_groups[i])))
                cl_id_2 = random.choice(list(group_to_clusters[g_cl_id_2].intersection(cluster_ids)))
                combinations = list(itertools.product(clusters[cl_id_1], clusters[cl_id_2]))
                sampling_ratio = 1 if len(clusters[cl_id_1]) + len(clusters[cl_id_2]) == 2 else 0.9
                num_samples = int(sampling_ratio * len(combinations))
                sampled_fp = set(random.sample(combinations, num_samples))
                missing_fp = num_fp - len(false_positives)
                false_positives = false_positives.union(set(random.sample(list(sampled_fp), min(missing_fp, num_samples))))
                queue = queue.union(sampled_fp.difference(false_positives))
                clusters[cl_id_1] = clusters[cl_id_1].union(clusters[cl_id_2])
                clusters[cl_id_2] = None
                cluster_ids.remove(cl_id_2)
                for g in list(cluster_to_groups[cl_id_2].difference(cluster_to_groups[cl_id_1])):
                    group_to_clusters[g].add(cl_id_1)
                cluster_to_groups[cl_id_1] = cluster_to_groups[cl_id_1].union(cluster_to_groups[cl_id_2])
                iter_id += 1
            candidate_sets.append(tp.union(false_positives))
            print("Candidates: %s, TP: %s, FP: %s, R: %s, P: %s, |C|: %s" % (len(candidate_sets[-1]), len(tp), len(false_positives),
                                                                             len(tp) / len(set_matches),
                                                                             len(tp) / len(candidate_sets[-1]),
                                                                             len([c for c in clusters if c is not None])))
            cand_df = pd.DataFrame(candidate_sets[-1], columns=["l_id", "r_id"])
            block_id += 1
            cand_df.to_csv("../blockers/candidates_synth_%s.csv" % (str(block_id) if len(str(block_id)) > 1
                                                                                  else "0" + str(block_id)), index=False)


if __name__ == "__main__":
    main()

path_dir_ds = "./datasets/"
file_ds = "dataset.csv"
file_gold = "matches.csv"
file_candidates = "blockers/candidates_"
file_blocks = "blockers/blocks_"
# string_null_value = "NULL"

datasets = {"alaska_cameras":
                {"path_ds": path_dir_ds + "alaska_cameras/" + file_ds,
                 "attributes": ["_id", "description", "brand", "model", "type", "mp",
                                "optical_zoom", "digital_zoom", "screen_size", "price"],
                 "time_attribute": None,
                 "default_aggregation": "vote",
                 "default_fusion":
                    {
                        "brand": "vote",
                        "model": "vote",
                        "type": "vote",
                        "mp": "vote",
                        "optical_zoom": "vote",
                        "digital_zoom": "vote",
                        "screen_size": "vote",
                        "price": "avg"
                    },
                 "blockers":
                    {
                        "None (Cartesian Product)":
                            {"path_candidates": None,
                             "path_blocks": None},
                        "SparkER (Meta-Blocking)":
                            {"path_candidates": path_dir_ds + "alaska_cameras/" + file_candidates + "sparker.pkl",
                             "path_blocks": path_dir_ds + "alaska_cameras/" + file_blocks + "sparker.pkl"},
                    },
                 "default_blocker": "SparkER (Meta-Blocking)",
                 "matchers":
                    {
                        "None (Dirty)":
                            {"path_gold": None},
                        "Ground Truth":
                            {"path_gold": path_dir_ds + "alaska_cameras/" + file_gold}
                    },
                 "default_matcher": "Ground Truth",
                 "pipelines":
                    {
                         "bf_sparker_mf_gt":
                            {"blocker": "SparkER (Meta-Blocking)",
                             "matcher": "Ground Truth"}
                    }
                },
            "nc_voters":
                {"path_ds": path_dir_ds + "nc_voters/" + file_ds,
                 "attributes": ["_id", "first_name", "last_name", "age", "birth_place", "sex", "race", "address",
                                "city", "zip_code", "street_name", "house_number", "party", "registration_date"],
                 "time_attribute": "registration_date",
                 "default_aggregation": "vote",
                 "default_fusion":
                    {
                        "first_name": "vote",
                        "last_name": "vote",
                        "age": "max",
                        "birth_place": "vote",
                        "sex": "vote",
                        "race": "vote",
                        "city": "vote",
                        "zip_code": "vote",
                        "party": "vote",
                        "registration_date": "min"
                    },
                 "blockers":
                    {
                        "None (Cartesian Product)":
                            {"path_candidates": None,
                             "path_blocks": None},
                        "PyJedAI (Similarity Join)":
                            {"path_candidates": path_dir_ds + "nc_voters/" + file_candidates + "pyjedai.pkl",
                             "path_blocks": path_dir_ds + "nc_voters/" + file_blocks + "pyjedai.pkl"},
                    },
                 "default_blocker": "PyJedAI (Similarity Join)",
                 "matchers":
                    {
                        "None (Dirty)":
                            {"path_gold": None},
                        "Ground Truth":
                            {"path_gold": path_dir_ds + "nc_voters/" + file_gold}
                    },
                 "default_matcher": "Ground Truth",
                 "pipelines":
                    {
                         "bf_pyjedai_mf_gt":
                            {"blocker": "PyJedAI (Similarity Join)",
                             "matcher": "Ground Truth"}
                    }
                },
            "nyc_funding_applications":
                {"path_ds": path_dir_ds + "nyc_funding_applications/" + file_ds,
                 "attributes": ["_id", "name", "address", "year", "agency", "source", "counselor", "amount", "status"],
                 "time_attribute": "year",
                 "default_aggregation": "vote",
                 "default_fusion":
                    {
                        "name": "vote",
                        "address": "vote",
                        "source": "vote",
                        "amount": "sum"
                    },
                 "blockers":
                     {
                        "None (Cartesian Product)":
                            {"path_candidates": None,
                             "path_blocks": None},
                         "SparkER (Meta-Blocking)":
                            {"path_candidates": path_dir_ds + "nyc_funding_applications/" + file_candidates + "sparker.pkl",
                             "path_blocks": path_dir_ds + "nyc_funding_applications/" + file_blocks + "sparker.pkl"},
                     },
                 "default_blocker": "SparkER (Meta-Blocking)",
                 "matchers":
                    {
                        "None (Dirty)":
                            {"path_gold": None},
                        "Ground Truth":
                            {"path_gold": path_dir_ds + "nyc_funding_applications/" + file_gold}
                    },
                 "default_matcher": "Ground Truth",
                 "pipelines":
                     {
                         "bf_sparker_mf_gt":
                             {"blocker": "SparkER (Meta-Blocking)",
                              "matcher": "Ground Truth"}
                     }
                 }
            }

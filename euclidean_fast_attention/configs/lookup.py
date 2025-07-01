# def default_num_train(split: str):
#     return num_train_lookup[split]


# def calculate_batch_configs(
#         config, split: str, capacity_multiplier: float = 1.1
# ):
#     max_num_graphs = config.trainer["max_num_graphs"]
#     cutoff = config.model.cutoff
#     num_nodes, avg_num_neighbors = molecular_graph_lookup[cutoff][split]
#     max_num_nodes = (max_num_graphs - 1) * num_nodes + 1
#     max_num_edges = (max_num_graphs - 1) * avg_num_neighbors * num_nodes
#     max_num_edges = int(max_num_edges * capacity_multiplier)
#     return max_num_nodes, max_num_edges


# def get_max_length(split: str):
#     return max_length_lookup[split]


max_length_lookup = {
    "ethanol": 10.0,  # max length in data
    "aspirin": 10.0,
    "toluene": 10.0,
    "uracil": 10.0,
    "naphthalene": 10.0,
    "salicylic": 10.0,
    "malonaldehyde": 10.0,
    "at_at": 22.0,
    "at_at_cg_cg": 24.0,
    "ac_ala3_nhme": 12.0,
    "dha": 16.0,
    "buckyball_catcher": 15.0,
    "double_walled_nanotube": 33.0,
    "stachyose": 14.0,
}

lookup_cutoff4 = {  # (number of atoms, avg. number of neighbors per atom = max # total neighbors / num_atoms)
    "ethanol": (9, 8),
    "aspirin": (21, 20),
    "toluene": (15, 14),
    "uracil": (12, 11),
    "naphthalene": (18, 17),
    "salicylic": (16, 15),
    "malonaldehyde": (9, 8),
    "at_at": (60, 16),
    "at_at_cg_cg": (118, 18),
    "ac_ala3_nhme": (42, 17),
    "dha": (56, 18),
    "buckyball_catcher": (148, 18),
    "double_walled_nanotube": (370, 25),
    "stachyose": (87, 22),
}

lookup_cutoff5 = { # (number of atoms, avg. # of neighbors per atom = max # total neighbors / num_atoms)
    "ethanol": (9, 8),  
    "aspirin": (21, 20),
    "toluene": (15, 14),
    "uracil": (12, 11),
    "naphthalene": (18, 17),
    "salicylic": (16, 15),
    "malonaldehyde": (9, 8),
    "at_at": (60, 26),
    "at_at_cg_cg": (118, 30),
    "ac_ala3_nhme": (42, 27),
    "dha": (56, 29),
    "buckyball_catcher": (148, 33),
    "double_walled_nanotube": (370, 45),
    "stachyose": (87, 36),
}

molecular_graph_lookup = {
    4: lookup_cutoff4, 
    5: lookup_cutoff5
}

num_train_lookup = {
    "ethanol": 1000,
    "aspirin": 1000,
    "toluene": 1000,
    "uracil": 1000,
    "naphthalene": 1000,
    "salicylic": 1000,
    "malonaldehyde": 1000,
    "at_at": 3000,
    "at_at_cg_cg": 2000,
    "ac_ala3_nhme": 6000,
    "dha": 8000,
    "buckyball_catcher": 600,
    "double_walled_nanotube": 800,
    "stachyose": 8000,
}

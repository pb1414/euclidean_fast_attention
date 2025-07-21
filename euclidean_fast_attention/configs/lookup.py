geometric_graph_lookup_cutoff4_md22 = {  # (number of atoms, avg. number of neighbors per atom = max # total neighbors / num_atoms)
    "AT_AT": (60, 16),
    "AT_AT_CG_CG": (118, 18),
    "AcAla3NHMe": (42, 17),
    "DHA": (56, 18),
    "buckyball_catcher": (148, 18),
    "nanotube": (370, 25),
    "stachyose": (87, 22),
}


geometric_graph_lookup_cutoff5_md22 = { # (number of atoms, avg. # of neighbors per atom = max # total neighbors / num_atoms)
    "AT_AT": (60, 26),
    "AT_AT_CG_CG": (118, 30),
    "AcAla3NHMe": (42, 27),
    "DHA": (56, 29),
    "buckyball_catcher": (148, 33),
    "nanotube": (370, 45),
    "stachyose": (87, 36),
}


geometric_graph_lookup_md22 = {
    4: geometric_graph_lookup_cutoff4_md22, 
    5: geometric_graph_lookup_cutoff5_md22
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


def get_avg_num_neighbors_md22(split: str, cutoff: int):
        num_and_neighs = md22_avg_num_neighbors_lookup[cutoff][split]
        _, neighs = num_and_neighs
        return neighs

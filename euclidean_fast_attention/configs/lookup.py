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


def get_avg_num_neighbors_md22(split: str, cutoff: int):
        num_and_neighs = geometric_graph_lookup_md22[cutoff][split]
        _, neighs = num_and_neighs
        return neighs

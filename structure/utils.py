"""
Utility functions for structure analysis
"""

entity_type_list = [
    "protein", "dna", "rna", "crystal", "polymer",
    "chain", 
    "molecule", "ion", "aminoacid residue", "nucleotide residue", "crystal unit cell",
    "atom"
]

mapping_residue_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

mapping_residue_3to1 = {key: value for value, key in mapping_residue_1to3.items()}

residue1_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

residue3_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


element_list = [
    "H", "He", 
    "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", 
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

element_list_uppercase = [element.upper() for element in element_list]


mapping_element_to_atom_mass = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.066,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.390,
    "Ga": 69.723,
    "Ge": 72.640,
    "As": 74.922,
    "Se": 78.960,
    "Br": 79.904,
    "Kr": 83.800,
    "Rb": 85.468,
    "Sr": 87.620,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.940,
    "Tc": 98.906,
    "Ru": 101.070,
    "Rh": 102.906,
    "Pd": 106.420,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.600,
    "I": 126.904,
    "Xe": 131.293,
}



mapping_aminoacid_residue_to_bond = {
    "ALA": [("N","CA"), ("CA","C"), ("C","O"), ("CA","CB")],

    "ARG": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD"),
            ("CD","NE"), ("NE","CZ"), ("CZ","NH1"), ("CZ","NH2")],

    "ASN": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","OD1"), ("CG","ND2")],

    "ASP": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","OD1"), ("CG","OD2")],

    "CYS": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","SG")],

    "GLN": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD"),
            ("CD","OE1"), ("CD","NE2")],

    "GLU": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD"),
            ("CD","OE1"), ("CD","OE2")],

    "GLY": [("N","CA"), ("CA","C"), ("C","O")],

    "HIS": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"),
            ("CG","ND1"), ("CG","CD2"),
            ("ND1","CE1"), ("CE1","NE2"), ("NE2","CD2")],

    "ILE": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG1"), ("CG1","CD1"),
            ("CB","CG2")],

    "LEU": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD1"), ("CG","CD2")],

    "LYS": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD"),
            ("CD","CE"), ("CE","NZ")],

    "MET": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","SD"), ("SD","CE")],

    "PHE": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"),
            ("CG","CD1"), ("CG","CD2"),
            ("CD1","CE1"), ("CE1","CZ"),
            ("CD2","CE2"), ("CE2","CZ")],

    "PRO": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"), ("CG","CD"), ("CD","N")],  # ring closure

    "SER": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","OG")],

    "THR": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","OG1"), ("CB","CG2")],

    "TRP": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"),
            ("CG","CD1"), ("CG","CD2"),
            ("CD1","NE1"), ("NE1","CE2"), ("CE2","CD2"),
            ("CE2","CZ2"), ("CZ2","CH2"), ("CH2","CZ3"),
            ("CZ3","CE3"), ("CE3","CD2")],

    "TYR": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG"),
            ("CG","CD1"), ("CG","CD2"),
            ("CD1","CE1"), ("CE1","CZ"), ("CZ","OH"),
            ("CD2","CE2"), ("CE2","CZ")],

    "VAL": [("N","CA"), ("CA","C"), ("C","O"),
            ("CA","CB"), ("CB","CG1"), ("CB","CG2")],
}


mapping_aminoacid_residue_to_bond_with_order = {
    "ALA": [("N","CA",1), ("CA","C",1), ("C","O",2), ("CA","CB",1)],

    "ARG": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD",1),
            ("CD","NE",1), ("NE","CZ",1), ("CZ","NH1",1), ("CZ","NH2",1)],

    "ASN": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","OD1",2), ("CG","ND2",1)],

    "ASP": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","OD1",1.5), ("CG","OD2",1.5)],

    "CYS": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","SG",1)],

    "GLN": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD",1),
            ("CD","OE1",2), ("CD","NE2",1)],

    "GLU": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD",1),
            ("CD","OE1",1.5), ("CD","OE2",1.5)],

    "GLY": [("N","CA",1), ("CA","C",1), ("C","O",2)],

    "HIS": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1),
            ("CG","ND1",1.5), ("CG","CD2",1.5),
            ("ND1","CE1",1.5), ("CE1","NE2",1.5), ("NE2","CD2",1.5)],

    "ILE": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG1",1), ("CG1","CD1",1),
            ("CB","CG2",1)],

    "LEU": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD1",1), ("CG","CD2",1)],

    "LYS": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD",1),
            ("CD","CE",1), ("CE","NZ",1)],

    "MET": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","SD",1), ("SD","CE",1)],

    "PHE": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1),
            ("CG","CD1",1.5), ("CD1","CE1",1.5), ("CE1","CZ",1.5),
            ("CZ","CE2",1.5), ("CE2","CD2",1.5), ("CD2","CG",1.5)],

    "PRO": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1), ("CG","CD",1), ("CD","N",1)],  # ring closure

    "SER": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","OG",1)],

    "THR": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","OG1",1), ("CB","CG2",1)],

    "TRP": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1),
            # indole (aromatic) ring system
            ("CG","CD1",1.5), ("CD1","NE1",1.5), ("NE1","CE2",1.5), ("CE2","CD2",1.5), ("CD2","CG",1.5),
            ("CE2","CZ2",1.5), ("CZ2","CH2",1.5), ("CH2","CZ3",1.5), ("CZ3","CE3",1.5), ("CE3","CD2",1.5)],

    "TYR": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG",1),
            ("CG","CD1",1.5), ("CD1","CE1",1.5), ("CE1","CZ",1.5),
            ("CZ","CE2",1.5), ("CE2","CD2",1.5), ("CD2","CG",1.5),
            ("CZ","OH",1)],

    "VAL": [("N","CA",1), ("CA","C",1), ("C","O",2),
            ("CA","CB",1), ("CB","CG1",1), ("CB","CG2",1)],
}



mapping_amino_acid_to_atom_names = {
    "ALA": ["N","CA","C","O","CB"],
    "ARG": ["N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2"],
    "ASN": ["N","CA","C","O","CB","CG","OD1","ND2"],
    "ASP": ["N","CA","C","O","CB","CG","OD1","OD2"],
    "CYS": ["N","CA","C","O","CB","SG"],
    "GLN": ["N","CA","C","O","CB","CG","CD","OE1","NE2"],
    "GLU": ["N","CA","C","O","CB","CG","CD","OE1","OE2"],
    "GLY": ["N","CA","C","O"],
    "HIS": ["N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2"],
    "ILE": ["N","CA","C","O","CB","CG1","CG2","CD1"],
    "LEU": ["N","CA","C","O","CB","CG","CD1","CD2"],
    "LYS": ["N","CA","C","O","CB","CG","CD","CE","NZ"],
    "MET": ["N","CA","C","O","CB","CG","SD","CE"],
    "PHE": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ"],
    "PRO": ["N","CA","C","O","CB","CG","CD"],
    "SER": ["N","CA","C","O","CB","OG"],
    "THR": ["N","CA","C","O","CB","OG1","CG2"],
    "TRP": ["N","CA","C","O","CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"],
    "TYR": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH"],
    "VAL": ["N","CA","C","O","CB","CG1","CG2"],
}


# Typical bond lengths in Ångström; keys use element symbols in alphabetical order.
mapping_element_to_standard_bond_length = {
    ("C","C", 1): 1.54,   # sp3–sp3 single
    ("C","C", 1.5): 1.39, # aromatic (benzene)
    ("C","C", 2): 1.34,   # alkene
    ("C","C", 3): 1.20,   # alkyne

    ("C","N", 1): 1.47,   # amine
    ("C","N", 1.5): 1.33, # peptide / aromatic C–N (partial double)
    ("C","N", 2): 1.28,   # imine
    ("C","N", 3): 1.15,   # nitrile

    ("C","O", 1): 1.43,   # alcohol/ether
    ("C","O", 1.5): 1.26, # carboxylate (COO–), two equal C–O
    ("C","O", 2): 1.23,   # carbonyl (C=O)

    ("N","O", 1): 1.44,   # hydroxylamine-like
    ("N","O", 2): 1.20,   # N=O

    ("C","S", 1): 1.82,   # thioether
    ("C","S", 2): 1.62,   # thioamide / C=S

    ("S","S", 1): 2.05,   # disulfide
}



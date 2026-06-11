# from __future__ import annotations
from .utils import *
import numpy as np
from .cif_reader import load_cif_to_data_dict, get_atom_site_data



class Structure:
    """
    General structure class
    """
    def __init__(
        self, 
        id: str,
        name: str,
    ):
        self.id = id
        self.name = name
        self.entity_type = "GeneralStructure"
        self.component_list = []
        self.metadata = {}

    def __repr__(self):
        return f"{self.entity_type}(id={self.id}, name={self.name})"

    def __getitem__(self, index):
        return self.component_list[index]

    def __len__(self):
        return len(self.component_list)

    def set_entity_type(self, entity_type: str):
        if entity_type in entity_type_list:
            self.entity_type = entity_type
        else:
            self.entity_type = "other"

    def add_component(self, component):
        if type(component) == list:
            self.component_list.extend(component)
        else:
            self.component_list.append(component)

    def get_atom_list(self):
        atom_list = []
        for component in self.component_list:
            if component.entity_type == "atom":
                atom_list.append(component)
            else:
                atom_list.extend(component.get_atom_list())
        return atom_list

    def set_property(self, key: str, value: any):
        self.metadata[key] = value
    
    def get_property(self, key: str):
        return self.metadata[key]
    
    def get_center_of_mass(self):
        return get_center_of_mass(self.get_atom_list())


class Atom(Structure):
    """
    Represents an atom with a name, element type, and 3D coordinates.
    """
    def __init__(
        self, 
        id: str,
        name: str, 
        element: str,
        coord_x: float, 
        coord_y: float, 
        coord_z: float,            
    ):
        super().__init__(id, name)
        self.set_entity_type("atom")
        self.element = element
        self.set_element(element)
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

    def set_element(self, element: str):
        if element in element_list_uppercase or element in element_list:
            self.element = element.capitalize()
        else:
            self.element = None
    
    def get_element(self):
        return self.element
    
    def get_atom_mass(self):
        return mapping_element_to_atom_mass[self.element]
    
    def get_coord(self):
        return np.array([self.coord_x, self.coord_y, self.coord_z])
    
    def set_coord(self, coord: np.array):
        self.coord_x = coord[0]
        self.coord_y = coord[1]
        self.coord_z = coord[2]



class Molecule(Structure):
    """
    Represents a molecule with a name and a list of atoms.

    Residues are molecules, ions are molecules, and so on.
    """
    def __init__(
        self, 
        id: str, 
        name: str,
    ):
        super().__init__(id, name)
        self.set_entity_type("molecule")
        
    def get_atom_by_name(self, name: str):
        for atom in self.component_list:
            if atom.name == name:
                return atom
        return None
    
    def get_atom_by_id(self, id: str):
        for atom in self.component_list:
            if atom.id == id:
                return atom
        return None
    
    def get_atom_list_by_element(self, element: str):
        atom_list = []
        for atom in self.component_list:
            if atom.element == element:
                atom_list.append(atom)
        return atom_list
    

class AminoAcidResidue(Molecule):
    """
    Represents an amino acid residue with a name and a list of atoms.
    """
    def __init__(
        self, 
        id: str, 
        name: str,
    ):
        super().__init__(id, name)
        self.set_entity_type("aminoacid residue")
        if name in residue3_list:
            self.residue3 = name
        else:
            self.residue3 = "UNK"
        self.residue1 = mapping_residue_3to1.get(name, "X")
        self.real_id = self.id

    def get_bond_list(self):
        if self.residue3 in residue3_list:
            bond_list = []
            for bond in mapping_aminoacid_residue_to_bond_with_order[self.residue3]:
                bond_list.append(Bond(self.get_atom_by_name(bond[0]), self.get_atom_by_name(bond[1]), bond[2]))
            return bond_list
        else:
            return []


class Chain(Structure):
    """
    Represents a chain with a name and a list of residues (molecules).
    """
    def __init__(
        self, 
        id: str, 
        name: str,
    ):
        super().__init__(id, name)
        self.set_entity_type("chain")
        
    def get_residue_by_id(self, id: str):
        for residue in self.component_list:
            if residue.id == id:
                return residue
        return None
    
    def get_residue_list_by_name(self, name: str):
        residue_list = []
        for residue in self.component_list:
            if residue.name == name:
                residue_list.append(residue)
        return residue_list
    

class Bond:
    """
    Represents a bond between two atoms.
    """
    def __init__(
        self, 
        atom1: Atom, 
        atom2: Atom,
        bond_order: float = 1
    ):
        atom_mass1 = atom1.get_atom_mass()
        atom_mass2 = atom2.get_atom_mass()
        if atom_mass1 > atom_mass2:
            self.atom1 = atom2
            self.atom2 = atom1
        else:
            self.atom1 = atom1
            self.atom2 = atom2
        self.bond_order = bond_order

    def __repr__(self):
        return f"Bond(atom1={self.atom1.name}, atom2={self.atom2.name}, bond_order={self.bond_order})"
    
    def get_atoms(self):
        return (self.atom1, self.atom2)
    
    def get_bond_length(self):
        return get_distance(self.atom1, self.atom2)
    
    def get_bond_order(self):
        return self.bond_order

    def get_standard_bond_length(self):
        standard_bond_length = mapping_element_to_standard_bond_length.get((self.atom1.element, self.atom2.element, self.bond_order), None)
        return standard_bond_length

    def get_bond_elements(self):
        return (self.atom1.element, self.atom2.element)

    def get_bond_type(self):
        return (self.atom1.element, self.atom2.element, self.bond_order)
    
    def get_center_of_mass(self):
        return get_center_of_mass([self.atom1, self.atom2])



def get_distance(atom1: Atom, atom2: Atom):
    return np.linalg.norm(atom1.get_coord() - atom2.get_coord())

def get_center_of_mass(atom_list: list[Atom]):
    return np.mean(get_atom_coord_array(atom_list) * get_atom_mass_array(atom_list).reshape(-1, 1), axis=0) / np.sum(get_atom_mass_array(atom_list))


def get_atom_coord_array(atom_list: list[Atom]):
    return np.array([atom.get_coord() for atom in atom_list])

def get_atom_mass_array(atom_list: list[Atom]):
    return np.array([atom.get_atom_mass() for atom in atom_list])

        
def load_cif(cif_file_path):
    data_dict = load_cif_to_data_dict(cif_file_path)
    atom_site_df = get_atom_site_data(data_dict)

    target = Structure(id="0", name=cif_file_path.split("/")[-1].split(".cif")[0])


    # initialize the first chain and residue
    fisrt_item = atom_site_df.iloc[0]
    first_chain_id = fisrt_item["_atom_site.label_entity_id"]
    first_chain_name = fisrt_item["_atom_site.label_asym_id"]
    first_residue_id = fisrt_item["_atom_site.label_seq_id"]
    first_residue_name = fisrt_item["_atom_site.label_comp_id"]
    first_model_num = fisrt_item["_atom_site.pdbx_PDB_model_num"]

    current_chain = Chain(id=first_chain_id, name=first_chain_name)
    current_residue = AminoAcidResidue(id=first_residue_id, name=first_residue_name)


    # store the chain, residue, and atom
    cache_chain_list = []
    cache_residue_list = []
    cache_atom_list = []


    for i in atom_site_df.index:
        item = atom_site_df.iloc[i]
        chain_id = item["_atom_site.label_entity_id"]
        chain_name = item["_atom_site.label_asym_id"]
        residue_id = item["_atom_site.label_seq_id"]
        residue_name = item["_atom_site.label_comp_id"]
        model_num = item["_atom_site.pdbx_PDB_model_num"]

        if model_num != first_model_num:
            continue    # only the first model is considered

        atom = Atom(
            id=item["_atom_site.id"], 
            name=item["_atom_site.label_atom_id"], 
            element=item["_atom_site.type_symbol"], 
            coord_x=float(item["_atom_site.Cartn_x"]), 
            coord_y=float(item["_atom_site.Cartn_y"]), 
            coord_z=float(item["_atom_site.Cartn_z"])
        )
        if residue_id != current_residue.id or residue_name != current_residue.name:
            current_residue.add_component(cache_atom_list)
            cache_atom_list = []
            cache_residue_list.append(current_residue)
            current_residue = AminoAcidResidue(id=residue_id, name=residue_name)

        if chain_id != current_chain.id or chain_name != current_chain.name:
            current_chain.add_component(cache_residue_list)
            cache_residue_list = []
            cache_chain_list.append(current_chain)
            current_chain = Chain(id=chain_id, name=chain_name)

        cache_atom_list.append(atom)

    # end of the loop
    current_residue.add_component(cache_atom_list)
    cache_atom_list = []
    cache_residue_list.append(current_residue)
    current_chain.add_component(cache_residue_list)
    cache_residue_list = []
    cache_chain_list.append(current_chain)

    target.add_component(cache_chain_list)

    return target



        





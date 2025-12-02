from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem.Draw import MolToImage
import pandas as pd
from urllib.request import urlopen
from urllib.parse import quote
import time
import math
import matplotlib.pyplot as plt
from itertools import combinations, chain, permutations
import random
import requests
import re

from more_itertools import split_when, pairwise
from itertools import chain
from collections import Counter


def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return None


def extract_iupac_smiles(iupac_oi):
    if not iupac_oi or str(iupac_oi).lower() == 'nan':
        return []

    delimiter = ';'
    split_iupac = [item.strip() for item in iupac_oi.split(delimiter)] if delimiter in iupac_oi else [iupac_oi.strip()]
    valid_smiles = []

    for name in split_iupac:
        time.sleep(1)
        try:
            smiles = CIRconvert(name)
            if smiles:
                smiles = smiles.strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    print(canonical_smiles)
                    valid_smiles.append(canonical_smiles)
        except Exception as e:
            print(f"Error converting {name}: {e}")
            continue

    if len(valid_smiles) > 0:
        print(f'The IUPAC-derived SMILES are: {valid_smiles}')

    return valid_smiles


def nest_brackets(tokens, i = 0):
    l = []
    while i < len(tokens):
        if tokens[i] == ')':
            return i,l
        elif tokens[i] == '(':
            i,subl = nest_brackets(tokens, i+1)
            l.append(subl)
        else:
            l.append(tokens[i])
        i += 1
    return i,l


def parse_compound(s):
    tokens = [''.join(t) for t in split_when(s, lambda a,b: b.isupper() or b in '()' or (b.isdigit() and not a.isdigit()))]
    tokens = [(int(t) if t.isdigit() else t) for t in tokens]
    i, l = nest_brackets(tokens)
    assert(i == len(tokens)) # crash if unmatched ')'
    return l


def count_elems(parsed_compound):
    c = Counter()
    for a,b in pairwise(chain(parsed_compound, (1,))):
        if not isinstance(a, int):
            subcounter = count_elems(a) if isinstance(a, list) else {a: 1}
            n = b if isinstance(b, int) else 1
            for elem,k in subcounter.items():
                c[elem] += k * n
    return c


def count_proper(formula_oi):
    l = parse_compound(formula_oi)
    c = count_elems(l)
    total_atoms = sum(c.values())

    return total_atoms


def filter_duplicates(structure_list):
    filtered_structure_list = []
    seen_structures = set()

    for structure in structure_list:
        if (
            structure and                      # not None or empty
            str(structure).strip().lower() != 'nan' and
            structure not in seen_structures
        ):
            filtered_structure_list.append(structure)
            seen_structures.add(structure)

    return filtered_structure_list


def get_cas_smiles_inchi(dataset):
    cas_list = dataset["CAS"].tolist()
    smiles_list = dataset["SMILES code"].tolist()
    formula_list = dataset["Molecular formula"].tolist()
    inchi_list = dataset["InChI Identifier"].tolist()
    qsar_smiles_list = dataset["Additional SMILES"].tolist()
    qsar_formula_list = dataset["Additional formula"].tolist()
    iupac_list = dataset["IUPAC name"].tolist()
    name_list = dataset["Substance name"].tolist()

    filtered_smiles_list = [item for item in smiles_list if str(item) != 'nan']
    filtered_inchi_list = [item for item in inchi_list if str(item) != 'nan']
    print(f'You have {len(filtered_smiles_list)} smiles to deal with')
    print(f'You have {len(filtered_inchi_list)} inchis to deal with')

    complete_list = []
    formulae_if_necessary = []

    for i, cas_oi in enumerate(cas_list):
        smiles_oi = str(smiles_list[i])
        formula_oi = str(formula_list[i])
        inchi_oi = str(inchi_list[i])
        add_formula_oi = str(qsar_formula_list[i])
        add_smiles_oi = str(qsar_smiles_list[i])
        iupac_oi = str(iupac_list[i])
        name_oi = str(name_list[i])

        print(f'\nNow processing {cas_list[i]} ({name_oi}). \nThis is UVCB {i + 1}/{len(cas_list)}.')

        fresh_smiles = sort_smiles(smiles_oi)
        fresh_inchi = sort_inchi(inchi_oi)
        add_fresh_smiles = sort_smiles(add_smiles_oi)
        fresh_iupac = extract_iupac_smiles(iupac_oi)

        fresh_formula = check_formula(smiles_oi, formula_oi)
        add_fresh_formula = check_formula(add_smiles_oi, add_formula_oi)

        complete_list = [*fresh_smiles, *fresh_inchi, *add_fresh_smiles, *fresh_iupac]
        formulae_if_necessary = [fresh_formula, add_fresh_formula]

        complete_list = filter_duplicates(complete_list)

        print(f'There are a total of {len(complete_list)} unique structures for {cas_oi}: {complete_list}')
        cc_carbon_lengths(complete_list)

    return complete_list, formulae_if_necessary


def check_formula(smiles_oi, formula_oi):
    equivalent = count_compare(smiles_oi, formula_oi)
    if not equivalent and str(formula_oi) != 'nan':
        print(f'The molecular formula, {formula_oi} is not associated with any given SMILES.')
        return formula_oi
    else:
        return None


def compute_carbon_length(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    def dfs(atom_idx, visited):
        visited.add(atom_idx)
        max_length = 1
        for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            n_idx = neighbour.GetIdx()
            atom = mol.GetAtomWithIdx(n_idx)
            if (
                n_idx not in visited and
                atom.GetSymbol() == 'C' and
                not atom.GetIsAromatic()
            ):
                sub_length = dfs(n_idx, visited.copy())
                if sub_length is not None:
                    max_length = max(max_length, 1 + sub_length)
        return max_length

    max_chain = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
            chain_length = dfs(atom.GetIdx(), set())
            if chain_length is not None:
                max_chain = max(max_chain, chain_length)


    return max_chain


def cc_carbon_lengths(smiles_list): # complile all
    length_list = []

    if len(smiles_list) != 0:
        for smiles in smiles_list:
            length = compute_carbon_length(smiles)
            if length > 0:
                length_list.append(length)
            else:
                continue
        carbon_range(length_list)
    else:
        print('No carbon range available')

    return length_list

def carbon_range(length_list):
    c_range = []
    if not length_list or length_list == 0:
        print('There are either no structures available or they are entirely aromatic')
    else:
        c_max = max(length_list)
        c_min = min(length_list)
        if c_max - c_min == 0:
            print(f'All structures provided have carbon numbers C{c_min}.')
            c_range = [c_min, c_max]
        else:
            print(f'These hydrocarbon structures range from C{c_min}-C{c_max}.')
            c_range = [c_min, c_max]

    return c_range

def count_compare(smiles_oi, formula_oi):
    equivalent = True

    if str(formula_oi) != 'nan':
        atom_count = count_proper(formula_oi)
        # print(f'we are working with {atom_count} atoms from the formula')
        try:
            mol = Chem.MolFromSmiles(smiles_oi)
            mol = Chem.AddHs(mol)
            smile_count = mol.GetNumAtoms()
           # print(f'We are working with {smile_count} atoms from the SMILES')
            if smile_count != atom_count:
                equivalent = False
        except Exception as e:
            return None
    else:
        return None

    return equivalent


def sort_smiles(smiles_oi):
    delimiter = '.'
    if not smiles_oi or str(smiles_oi) == 'nan':
        return []

    split_smiles = smiles_oi.split(delimiter) if delimiter in smiles_oi else [smiles_oi]
    print(f'The existing SMILES are: {split_smiles}')
    return split_smiles


def sort_inchi(inchi_oi):
    delimiter = '.'
    split_inchi_smiles = []

    if not inchi_oi or str(inchi_oi) == 'nan':
        return []

    try:
        mol = Chem.MolFromInchi(inchi_oi)
        if mol is not None:
            inchi_smiles = Chem.MolToSmiles(mol, canonical=True)
            if delimiter in inchi_smiles:
                split_inchi_smiles = inchi_smiles.split(delimiter)
            else:
                split_inchi_smiles = [inchi_smiles]
        else:
            print("Invalid InChI: could not generate molecule.")
    except Exception as e:
        print(f"Error during InChI processing: {e}")

    print(f"The InChi-derived SMILES are: {split_inchi_smiles}")
    return split_inchi_smiles


if __name__ == "__main__":
    import requests

    name = "ethanol"
    url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(name)}/smiles"
    response = requests.get(url)
    print(response.text)

    compiled_dataset = pd.read_excel('House_data_compiled.xlsx')
    test_list = ['CCCCCCC1=CC=C2C(C=CC3=CC(CCCCCC(C)CCCC)=CC=C23)=C1', 'CCCCCCCCCCC(CCCC)CCCCCCCCCCCCCCCCC(CCCC)(CCCC)CCCCCCCC', 'CCCCCCC1=CC2=C3C4C5C(SC6=C5C5=C3C3=C7C(C=C(CCCCCC(C)CCC)N=C7C7CC8CCCCC8C8=C7C3=C2C2=CC(O)=CC=C82)=C5C=C6)=C(C)C=C14', 'CCCCCCC1=Cc2c3c4c5c6c(ccc5c5cc(CCCCCC(C)CCC)nc7c5c4c4c5c(c8ccc(O)cc8c24)C2CCCCC2CC75)SC2=C(C)C=C1C3C26', 'CCCCCCc1ccc2c(ccc3cc(CCCCCC(C)CCCC)ccc32)c1']
    cc_carbon_lengths(test_list)
    get_cas_smiles_inchi(compiled_dataset)


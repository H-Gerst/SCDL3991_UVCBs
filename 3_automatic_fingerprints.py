import os
import csv
import pandas as pd
from fontTools.misc.classifyTools import Classifier
from rdkit import Chem
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.ticker import MaxNLocator

# === Load SMARTS definitions ===
def load_smarts_with_numbers(file_path, sheet_name="Sheet1"):

    smart_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    smarts_data = []

    for idx, row in smart_df.iterrows():
        row_number = idx + 1  # Excel-style row number
        classifier = str(row[1]).strip() if not pd.isna(row[1]) else None
        if not classifier:
            continue

        smarts_list = [str(x).strip() for x in row[2:].dropna() if "[" in str(x)]
        for s in smarts_list:
            mol = Chem.MolFromSmarts(s)
            if mol:
                smarts_data.append({
                    "row": row_number,
                    "classifier": classifier,
                    "smarts": s,
                    "mol": mol
                })
            else:
                print(f"Invalid SMARTS for group '{classifier}' (row {row_number}): {s}")

    print(f"Loaded {len(smarts_data)} valid SMARTS in total.")
    return smarts_data

# === Helper functions ===
def count_atoms(mol):
    carbons = hydrogens = heteroatoms = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "C":
            carbons += 1
        else:
            heteroatoms += 1

    hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    return carbons, hydrogens, heteroatoms

def check_double_bond(mol):
    db = False
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.AROMATIC:
            db = True
            return db

    return db

def count_double_bond(mol):
    double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == rdchem.BondType.DOUBLE)
    return double_bonds

def check_any_branch(mol):
    br = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon_neighbours = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
            if carbon_neighbours >= 3:
                br = True

    return br

def check_aromaticity(mol):
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)

    aromatic = False
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic = True
            return aromatic

    return aromatic
    #return any(a.GetIsAromatic() for a in mol.GetAtoms())

def check_cyclisation(mol):
    ring = False
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            ring = True
            return ring
    return ring


def has_substituted_cycloalkenes(mol):
    rings = mol.GetRingInfo().AtomRings()

    for ring in rings:
        # Check for non-aromatic double bond
        has_double_bond = any(
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType() == Chem.rdchem.BondType.DOUBLE
            and not mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetIsAromatic()
            for i in range(len(ring))
        )
        if not has_double_bond:
            continue

        # Check for substitution
        substituted = any(
            any(nbr.GetIdx() not in ring for nbr in mol.GetAtomWithIdx(idx).GetNeighbors())
            for idx in ring
        )

        if substituted:
            return True

    return False

def count_alkyl_branches(mol):
    branches = 0
    for atom in mol.GetAtoms():
        # Only consider carbons that are not in a ring
        if atom.GetAtomicNum() == 6 and not atom.IsInRing() and atom.GetDegree() >= 3:
            # Check neighbors that are also not in a ring
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6 and not nbr.IsInRing():
                    branches += 1
    return branches

def ring_counter(mol):
    return mol.GetRingInfo().NumRings()

def has_large_fused_pah(mol, min_rings):
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    # Extract aromatic rings
    aromatic_rings = [set(r) for r in atom_rings
                      if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)]

    if len(aromatic_rings) < min_rings:
        return False

    # Build adjacency map of fused rings
    adjacency = {i: set() for i in range(len(aromatic_rings))}
    for i, r1 in enumerate(aromatic_rings):
        for j, r2 in enumerate(aromatic_rings):
            if i < j and len(r1.intersection(r2)) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Depth-first search to find connected fused systems
    visited = set()
    def dfs(ring_idx, cluster):
        visited.add(ring_idx)
        cluster.add(ring_idx)
        for nbr in adjacency[ring_idx]:
            if nbr not in visited:
                dfs(nbr, cluster)

    # Find all fused aromatic clusters
    fused_clusters = []
    for i in range(len(aromatic_rings)):
        if i not in visited:
            cluster = set()
            dfs(i, cluster)
            fused_clusters.append(cluster)

    # Check if any cluster has at least min_rings fused
    return any(len(cluster) >= min_rings for cluster in fused_clusters)

def count_aromatic_rings(mol):
    # Ensure aromaticity perception
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    aromatic_ring_count = 0

    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_ring_count += 1

    return aromatic_ring_count

def check_metal(mol):
    metal = False
    metal_atomic_numbers = [
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112
    ]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in metal_atomic_numbers:
            metal = True
            return metal
        else:
            continue
    return metal

def check_non_metal(mol):
    non_metal = False
    non_metal_numbers = [7, 8, 16]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in non_metal_numbers:
            non_metal = True
            return non_metal

    return non_metal

def has_alkyl_substituted_aromatic_ring(mol):
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
            # check neighbors that are non-aromatic carbon atoms
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C' and not nbr.GetIsAromatic() and nbr.GetHybridization() == Chem.HybridizationType.SP3:
                    return True
    return False

def has_noncarbon_substituted_aromatic_ring(mol):
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
            # Check neighbors that are non-aromatic
            for nbr in atom.GetNeighbors():
                # If neighbor is non-aromatic AND not carbon → counts as hetero substitution
                if not nbr.GetIsAromatic() and nbr.GetSymbol() != 'C':
                    return True
    return False

def count_noncarbon_substituents_aromatic_ring(mol):
    # Ensure aromaticity perception is up-to-date
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)

    count = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
            for nbr in atom.GetNeighbors():
                # Count neighbors that are non-aromatic and NOT carbon
                if not nbr.GetIsAromatic() and nbr.GetSymbol() != 'C':
                    count += 1
    return count

def count_alkyl_substituents_aromatic_ring(mol):
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C' and not nbr.GetIsAromatic() and nbr.GetHybridization() == Chem.HybridizationType.SP3:
                    count += 1
    return count

def has_double_bond_in_hydrocarbon_ring(mol):
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        # Check if all atoms are carbon
        if all(atom.GetSymbol() == 'C' for atom in ring_atoms):
            # Check for a C=C double bond within that ring
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in ring and
                    bond.GetEndAtomIdx() in ring and
                    bond.GetBondType() == Chem.rdchem.BondType.DOUBLE):
                    return True
    return False

def classify_heteroatoms(mol, element):
    results = []
    ptable = Chem.GetPeriodicTable()
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    element_name = ptable.GetElementName(element)
    smarts_file = load_smarts_with_numbers("Structures_for_decision.xlsx", sheet_name="Sheet2")
    classifiers_oi = [90,91,92,93,94,95,96,]

    for atom in mol.GetAtoms():
        current_results = []

        if atom.GetAtomicNum() == element:
            current_results.append(element_name)
            acid_classification = classify_smiles(mol, classifiers_oi, smarts_file)
            current_results = current_results + acid_classification

            if atom.GetIsAromatic():
                classification = f"Aromatic {element_name}"
                current_results.append(classification)
            elif atom.IsInRing():
                classification = f"Non-aromatic heterocyclic {element_name}"
                current_results.append(classification)
            else:
                current_results.append(f"Aliphatic {element_name}")

            results = results + current_results
            continue
    #print(results)
    return results


# ======= Classification process =======
def smiles_tree(smiles, smarts_file):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if not mol:
        return {
            'SMILES': smiles,
            "Total Carbons": 0,
            "Total Hydrogens": 0,
            "Total Heteroatoms": 0,
            "Classifications": "Invalid SMILES"
        }

    classifiers_oi = []
    classifications = []
    has_double_bond = check_double_bond(mol)
    double_bond_count = count_double_bond(mol)
    alkyl_branches = count_alkyl_branches(mol)
    cyclised = check_cyclisation(mol)
    ring_count = ring_counter(mol)
    aromatic = check_aromaticity(mol)

    aromatic_count = count_aromatic_rings(mol)
    substituted_cycloalkenes = has_substituted_cycloalkenes(mol)

    carbons, hydrogens, heteroatoms = count_atoms(mol)

    if heteroatoms > 0:
        if check_metal(mol): # 1 Contains metal heteroatom(s)
            classifications.append("Contains metal heteroatom(s)")
            classifiers_oi = [2,3,4,5,6,7,8,9,10,11,12] # Arsenic, barium, cadmium, chromium, lead, zinc, sodium, nickel, iron, vanadium, iodine
            classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
        if check_non_metal(mol): # 13 Contains non-metal heteroatom(s)
            classifications.append("Contains non-metal heteroatom(s)")
            classifiers_oi = [18,21,24] # Aromatic nitrogen, aromatic sulfur, aromatic oxygen
            classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
            classifications = classifications + classify_heteroatoms(mol, 7) # Classifiers 17,18,19
            classifications = classifications + classify_heteroatoms(mol, 16) # Classifiers 20,21,22
            classifications = classifications + classify_heteroatoms(mol, 8) # Classifiers 23,24,25

    elif not aromatic:
        if check_any_branch(mol):
            classifications.append("Any branch") # 33
        if not has_double_bond:
            classifications.append("Saturated aliphatic") # 26
            if not cyclised:
                if alkyl_branches == 0:
                    classifications.append("Straight chain alkane/paraffin") # 27
                    classifiers_oi = [28,29,30,31,32,84] # C>18, C18-30, C40-55, C56-80, methane
                    classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                else:
                    classifications.append("Branched alkane/iso-paraffin") # 34
                    classifiers_oi = [35,36,37,38] # methyl branch, ethyl branch, C3-14 branch, neopentane
                    classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                    if alkyl_branches > 1:
                        classifications.append("Highly branched alkane/iso-paraffin+") # 39
            elif cyclised and ring_count != 0:
                classifications.append("Cycloalkane/cycloparaffin/naphthene") # 40
                if ring_count == 1:
                    classifications.append("Monocyclic alkane/monocyclic paraffin") # 41
                    classifiers_oi = [42,43] # cyclopentane, cyclohexane
                    classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                if ring_count >= 1:
                    classifications.append("Polycyclic alkane/polycyclic paraffin") # 44
                    classifiers_oi = [45,46,47,48,49,50] # Fused cyclopentanes, fused cyclohexanes, fused cyclopentane/cyclohexane, tricyclic terpane, hopane, sterane
                    classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                classifiers_oi = [51,52,53] # Substituted cycloalkane, alkyl-cyclopentane, alkyl-cyclohexane
                classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)

        elif has_double_bond:
            classifications.append("Unsaturated aliphatic") # 54
            if double_bond_count == 1:
                classifications.append("Mono-olefin")  # 56
            if double_bond_count > 1:
                classifications.append("Polyolefin")  # 57
            if not cyclised:
                if alkyl_branches == 0:
                    classifications.append("Straight chain alkene/olefin") # 51
                if alkyl_branches == 1:
                    classifications.append("Branched olefin/iso-olefin") # Classifier 58
                if alkyl_branches > 1:
                    classifications.append("Highly branched olefin/iso-olefin+") # Classifier 59
            elif cyclised and ring_count != 0:
                if has_double_bond_in_hydrocarbon_ring(mol): #double bond in ring:
                    classifications.append("Cycloalkene") # 60
                    classifiers_oi = [62,63,89]  # Cyclohexene, cyclopentene, dicyclopentadiene
                    classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                    if ring_count == 1:
                        classifications.append("Monocyclic alkene") # 61
                    if ring_count > 1:
                        classifications.append("Polycyclic alkene") # 67
                        classifiers_oi = [62,63]  # Cyclohexene, cyclopentene
                        classifications = classifications + classify_smiles(mol, classifiers_oi, smarts_file)
                    if substituted_cycloalkenes:
                        classifications.append("Substituted cycloalkene") # 64
                        classifiers_oi = [65,66] # Substituted cyclopentene, substituted cyclohexane
                        classifications = classifications +  classify_smiles(mol, classifiers_oi, smarts_file)

    elif aromatic:
        classifications.append("Aromatic") # 68
        if aromatic_count == 1:
            classifications.append("Monoaromatic") # 69
            if has_alkyl_substituted_aromatic_ring(mol):
                classifications.append("Alkyl-substituted monoaromatic") # 70
                if count_alkyl_substituents_aromatic_ring(mol) == 1:
                    classifications.append('Mono-alkyl-substituted monoaromatic') # 71
                if count_alkyl_substituents_aromatic_ring(mol) > 1:
                    classifications.append('Poly-alkyl-substituted monoaromatic') # 72
            else:
                classifications.append('Unsubstituted monoaromatic') # 73
        else:
            classifications.append('Polyaromatic') # 74
            if aromatic_count != ring_count:
                classifications.append("Mixed aromatic and non-aromatic-polycycle") # 75
            if has_large_fused_pah(mol, 2) and aromatic_count == 2: # add ring counter? to make sure it isn't reading aro and non aro as naphthalene
                classifications.append("Naphthalene") # 76
            if has_large_fused_pah(mol, 3) and aromatic_count == 3:
                classifications.append("Anthracene/phenanthrene") # 77
            if has_large_fused_pah(mol, 4) and aromatic_count == 4:
                classifications.append("Pyrene/chrysene/naphthacene/tetracene/benzanthracene") # 78
            if has_large_fused_pah(mol, 5) and ring_count == 5:
                classifications.append("5-ring PAH") # 79
            if has_large_fused_pah(mol, 6) and aromatic_count == 6:
                classifications.append("6-ring PAH") # 80
            if has_large_fused_pah(mol, 6) and aromatic_count >= 7:
                classifications.append("Heavy PAH (7+ rings)") # 81
            if has_alkyl_substituted_aromatic_ring(mol):
                classifications.append("Alkyl-substituted PAH") # 82
                if count_alkyl_substituents_aromatic_ring(mol) == 1:
                    classifications.append("Mono-alkyl-substituted PAH") # 83
                if count_alkyl_substituents_aromatic_ring(mol) > 1:
                    classifications.append("Poly-alkyl-substituted PAH") # 84
            if has_noncarbon_substituted_aromatic_ring(mol):
                if count_noncarbon_substituents_aromatic_ring(mol) == 1:
                    classifications.append("Mono-hetero-substituted PAH") # 85
                if count_noncarbon_substituents_aromatic_ring(mol) > 1:
                    classifications.append("Poly-hetero-substituted PAH") # 86
            else:
                classifications.append("Unsubstituted PAH") # 87

    unique_classifications = set(classifications)

    return {'SMILES' : smiles,
            "Total Carbons": carbons,
            "Total Hydrogens": hydrogens,
            "Total Heteroatoms": heteroatoms,
            "Classifications": "; ".join(unique_classifications)}, unique_classifications


def classify_smiles(mol, classifiers, smarts_file):
    selected_smarts = [d for d in smarts_file if d["row"] in classifiers]
    classifiers = []

    for sdata in selected_smarts:
        match = mol.HasSubstructMatch(sdata["mol"])
        if match:
            #print(sdata["classifier"])
            classifiers.append(sdata["classifier"])
        else:
            continue

    return classifiers

# === Detect and read SMILES row ===
def detect_delimiter(sample, guess_list=[',', '\t', ';', '|']):
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=guess_list)
        return dialect.delimiter
    except Exception:
        for d in guess_list:
            if d in sample:
                return d
        return ','

def read_smiles_row(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = detect_delimiter(sample)
        reader = csv.reader(f, delimiter=delimiter)

        for idx, row in enumerate(reader):
            row = [cell.strip() for cell in row if cell.strip()]
            if not row:
                continue
            if any(cell.lower().startswith("smiles") for cell in row):
                smiles_list = [cell for cell in row if not cell.lower().startswith("smiles")]
                if smiles_list:
                    print(f"Found SMILES row (line {idx+1}), {len(smiles_list)} SMILES detected.")
                    return smiles_list

    raise ValueError(f"Could not find a 'SMILES' row with data in {file_path}.")

def remove_duplicates(item_list):
    unique_items = []
    for item in item_list:
        if item:
            if item not in unique_items:
                unique_items.append(item)
            else:
                continue
    return unique_items

def count_smiles_per_cas(cas_numbers):
    cas_smiles_counts = []

    for cas_number in cas_numbers:
        input_file = f"{cas_number}_structures_extracted.csv"
        if not os.path.exists(input_file):
            continue

        try:
            smiles_list = read_smiles_row(input_file)
            n_smiles = len(smiles_list)
            cas_smiles_counts.append({"CAS": cas_number, "Num_SMILES": n_smiles})
        except Exception as e:
            print(f"Failed to read SMILES for {cas_number}: {e}")
            continue

    df_counts = pd.DataFrame(cas_smiles_counts)
    return df_counts


def summarize_smiles_counts(df_counts):
    summary = df_counts['Num_SMILES'].value_counts().sort_index()
    summary_df = pd.DataFrame({
        'Num_SMILES': summary.index,
        'Num_CAS': summary.values
    })
    print('food')
    print(summary_df)
    return summary_df

def plot_smiles_distribution_2(summary_df, title="Distribution of SMILES per CAS"):
    plt.figure(figsize=(26, 20))
    ax = sns.barplot(
        x='Num_SMILES',
        y='Num_CAS',
        data=summary_df,
        palette="tab20"
    )

    print(summary_df)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine(top=True, right=True)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    plt.tick_params(axis='both', length=20, width=3)  # match previous thickness
    plt.xlabel("Number of SMILES available", fontsize=60, labelpad=30)
    plt.ylabel("Number of UVCBs", fontsize=60, labelpad=30)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.grid(False)

    plt.tight_layout()
    plt.show()



import pandas as pd

def count_cas_with_smiles_per_category(cas_numbers, cas_categories, cas_smiles_counts):
    # Convert cas_smiles_counts to DataFrame
    df_smiles_counts = pd.DataFrame(cas_smiles_counts)

    # Combine CAS numbers with their categories
    category_df = pd.DataFrame({
        "CAS": cas_numbers,
        "Category": cas_categories
    })
    # Merge category info with SMILES counts
    merged_df = pd.merge(category_df, df_smiles_counts, on="CAS", how="left")

    # Filter only CAS with at least one SMILES
    merged_df = merged_df[merged_df['Num_SMILES'] > 0]

    # Count unique CAS per category
    category_counts = merged_df.groupby('Category')['CAS'].nunique().reset_index()
    category_counts.columns = ['Category', 'Num_CAS_with_SMILES']

    print(category_counts)

    return category_counts

def plot_smiles_distribution(x, y, title="Distribution of manual classifiers per CAS"):
    plt.figure(figsize=(30, 18))

    ax = sns.barplot(
        x=x,
        y=y,
        palette="tab20"
    )

    # Y-axis as integer
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Remove top/right spines, keep bottom/left and thicken
    sns.despine(top=True, right=True)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    # Styling
    plt.tick_params(axis='both', length=20, width=3)  # match previous thickness
    plt.xlabel("Number of automatically assigned classifiers", fontsize=60, labelpad=30)
    plt.ylabel("Number of UVCBs", fontsize=60, labelpad=30)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.grid(False)

    plt.tight_layout()
    plt.show()
    # sns.barplot(
    #     x=x,
    #     y=y,
    #     palette="tab20"
    # )
    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # # Styling
    # plt.tick_params(axis='both', length=15)
    # plt.xlabel("Number of automatically assigned classifiers", fontsize=55, labelpad=22)
    # plt.ylabel("Number of UVCBs", fontsize=55, labelpad=22)
    # plt.xticks(fontsize=50)
    # plt.yticks(fontsize=50)
    # plt.title("", fontsize=26)
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    all_smiles = []
    all_classifiers = []
    classifiers_per_uvcb = []
    smarts_structures = load_smarts_with_numbers("Structures_for_decision.xlsx", sheet_name="Sheet2")
    cas_df = pd.read_excel("House_data_compiled.xlsx")
    cas_numbers = cas_df.iloc[:, 0].dropna().astype(str).tolist()
    cas_categories = cas_df.iloc[:, 2].dropna().astype(str).tolist()

    cas_smiles_counts = []

    for cas_number in cas_numbers:
        input_file = f"{cas_number}_structures_extracted.csv"
        output_file = f"{cas_number}_composition_analysis.csv"

        if not os.path.exists(input_file):
            print(f"Missing file: {input_file}")
            continue

        try:
            smiles_list = read_smiles_row(input_file)
            cas_smiles_counts.append({"CAS": cas_number, "Num_SMILES": len(smiles_list)})
        except Exception as e:
            print(f"Failed to read SMILES for {cas_number}: {e}")
            continue

        analysis = []
        current_classifiers = []
        for smi in smiles_list:
            result, found_classes = smiles_tree(smi, smarts_structures)
            all_smiles.append(smi)
            all_classifiers.extend(found_classes)
            current_classifiers.extend(found_classes)
            if result:
                analysis.append(result)

        classifiers_per_uvcb.append(len(set(current_classifiers)))

        if not analysis:
            print(f"No valid SMILES found for {cas_number}")
            continue

        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "SMILES", "Total Carbons", "Total Hydrogens", "Total Heteroatoms", "Classifications"
            ])
            writer.writeheader()
            writer.writerows(analysis)

    unique_smiles = remove_duplicates(all_smiles)
    unique_classifiers = remove_duplicates(all_classifiers)
    print(len(unique_classifiers))
    print(len(classifiers_per_uvcb))

    counts = Counter(classifiers_per_uvcb)

    x = sorted(counts.keys())
    y = [counts[k] for k in x]
    plot_smiles_distribution(x, y)


    # Save counts
    df_smiles_counts = pd.DataFrame(cas_smiles_counts)
    df_smiles_counts.to_csv("CAS_smiles_counts.csv", index=False)

    summary_smiles = df_smiles_counts['Num_SMILES'].value_counts().sort_index().reset_index()
    summary_smiles.columns = ['Num_SMILES', 'Num_CAS']
    summary_smiles.to_csv("SMILES_distribution_summary.csv", index=False)


    category_counts = count_cas_with_smiles_per_category(cas_numbers, cas_categories, cas_smiles_counts)
    # Plot the distribution
    #plot_smiles_distribution(summary_smiles)
    plot_smiles_distribution_2(summary_smiles)




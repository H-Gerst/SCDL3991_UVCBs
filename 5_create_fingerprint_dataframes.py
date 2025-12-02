import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from adjustText import adjust_text
from tabulate import tabulate

total_classifier_list = ['Anthracene/Phenanthrene', 'Poly-alkyl-substituted PAH', 'Unsubstituted PAH', 'Saturated branch between C3-C14', 'Neopentane', 'Highly-branched alkane/Iso-paraffin+', 'Aromatic Nitrogen', 'Phenol', 'Carboxylic acids', 'Mixed aromatic and non-aromatic-polycycle', 'Heavy PAH (7+ rings)', 'Methane', 'C<18', 'C5-C40', 'Aliphatic Oxygen', '5-ring PAH)', 'Mono-cyclic alkane/Mono-cyclic paraffin', 'Cyclohexane', 'Substituted cycloalkane', 'C18-C30', 'Methyl branch', 'C40-C55', 'C56-C80', 'Ethyl branch', 'Aliphatic Nitrogen', 'Sodium', 'Iodine', 'Sulfonic acid', 'Naphthalene', '6-ring PAH)', 'Poly-alkyl-substituted mono-aromatic', 'Unsubstituted mono-aromatic', 'Mono-alkyl-substituted mono-aromatic', 'Mono-olefin', 'Straight chain alkene/Olefin', 'Mono-alkyl-substituted PAH', 'Cyclopentane', 'Fused cyclopentanes', 'Aliphatic Sulfur']
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)      # prevents line wrapping
pd.set_option("display.max_colwidth", None)

#  Summarizing to make fingerprint csv
def summarize_cas_files(
        cas_list,
        master_classifier_list,
        manual_list,
        sts_df,
        stats_to_include=["concawe_comp", "concawe_range", "chem", "pah", "none"],
        classifiers_from=["smiles", "manual", "none"]
):
    summary_rows = []
    missing_cas = []

    # which stats to return
    concawe_comp_fields = [
        'Carbon content',
        'Hydrogen content', 'Oxygen content',
        'Nitrogen content', 'Sulphur content'
    ]

    concawe_range_fields = ['Min carbon','Max carbon']

    chem_fields = ['Min carbon', 'Max carbon']

    pca_fields = [
        'total.Wt.Percent', 'P3_7_index', 'P4_7_index', 'P5_7_index',
        'P1_2_index', 'P1_index', 'P2_index', 'P3_index',
        'P4_index', 'P5_index', 'P6_index', 'P7_index',
        'Primary_PAH'
    ]

    all_manual_cls = []
    all_auto_cls = []

    for i, cas in enumerate(cas_list):
        row = {"CAS": cas}

        # Concawe stats
        if "concawe_comp" in stats_to_include:
            stats = get_concawe_stats(sts_df, cas)
            if (not stats) or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in stats.values()):
                #print(f"Dropping CAS {cas} due to incomplete stats: {stats}")
                continue
            for key in concawe_comp_fields:
                row[key] = stats.get(key)

        # Concawe stats
        if "concawe_range" in stats_to_include:
            range = get_concawe_range(sts_df, cas)
            if (not range) or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in range.values()):
                #print(f"Dropping CAS {cas} due to incomplete stats: {range}")
                missing_cas.append(cas)
                continue
            for key in concawe_range_fields:
                row[key] = range.get(key)

        # SMILES / carbon-hetero stats
        if "chem" in stats_to_include:
            input_file = f"{cas}_composition_analysis.csv"
            if os.path.exists(input_file):
                df = pd.read_csv(input_file)
                if not df.empty:
                    chem_stats = compute_carbon_heteroatom(df)
                    if (not chem_stats) or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in chem_stats.values()):
                 #       print(f"Dropping CAS {cas} due to incomplete stats: {chem_stats}")
                        missing_case.append(cas)
                        continue

                    for key in chem_fields:
                        row[key + "_chem"] = chem_stats.get(key)
                else:
                  #  print(f'Dropping CAS {cas} because composition file is empty')
                    missing_cas.append(cas)
                    continue
            else:
                print(f'Dropping CAS {cas} because composition file is empty')
                missing_cas.append(cas)
                continue

        # PCA stats
        if "pah" in stats_to_include:
            pca_stats = get_pca_stats(cas)
            if (not pca_stats) or all(v == 0 for v in pca_stats.values()):
                missing_cas.append(cas)
                continue
            for key in pca_fields:
                row[key] = pca_stats.get(key)


        classifiers = []

        if "none" in classifiers_from:
            summary_rows.append(row)
        else:
            # SMILES classifiers
            if "smiles" in classifiers_from:
                input_file = f"{cas}_composition_analysis.csv"
                if os.path.exists(input_file):
                    df = pd.read_csv(input_file)
                    if not df.empty:
                        smiles_classifiers = extract_unique_classifiers(df)
                        classifiers.extend(smiles_classifiers)
                        all_auto_cls.extend(smiles_classifiers)
                    else:
                        missing_cas.append(cas)
                        continue

            # Manual classifiers
            if "manual" in classifiers_from:
                manual_cls, present = extract_manual_classifiers(manual_list[i], all_fp_classifiers_df)
                if present:
                    classifiers.extend(manual_cls)
                    all_manual_cls.extend(manual_cls)


            row.update(
                {cl: 1 if cl in classifiers else 0 for cl in master_classifier_list}
            )

            if len(classifiers) != 0:
                summary_rows.append(row)
                #print(row)
            else:
                missing_cas.append(cas)

    final_auto_cls = set(all_auto_cls)
    final_manual_cls = set(all_manual_cls)

    df = pd.DataFrame(summary_rows)
    print(f"There are {len(summary_rows)} fingerprints")
    return df, missing_cas #

# Classifier extractions
def extract_unique_classifiers(df):
    SMILES_classifiers = []
    for entry in df['Classifications'].dropna():
        split = [c.strip() for c in entry.split(';') if c.strip()]
        SMILES_classifiers.extend(split)
    return sorted(set(SMILES_classifiers))

def extract_manual_classifiers(manual_descriptor_list, all_classifiers):
    manual_classifiers = []
    mcp = False

    if isinstance(manual_descriptor_list, int):
        if manual_descriptor_list != 0:
            try:
                classifier_name = all_classifiers.iloc[manual_descriptor_list]['Name']
                manual_classifiers.append(classifier_name)
                mcp = True
            except Exception:
                pass

    elif isinstance(manual_descriptor_list, str):
        try:
            split_mdl = [item.strip() for item in manual_descriptor_list.split(',')]
            for item in split_mdl:
                idx = int(item)
                classifier_name = all_classifiers.iloc[idx]['Name']
                manual_classifiers.append(classifier_name)
                mcp = True
        except:
            pass

    return manual_classifiers, mcp

def encode_classifier_presence(found_classifiers, master_classifier_list):
    found_set = set(found_classifiers)
    print(len(found_set))
    return [1 if classifier in found_set else 0 for classifier in master_classifier_list]

def get_concawe_stats(stats_df, cas):
    row = stats_df.loc[stats_df['CAS'] == cas]

    if row.empty:
        return None  # CAS not found → drop later

    row = row.iloc[0]

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    stats = {
        'Carbon content': safe_float(row.get('Carbon (% m/m)')),
        'Hydrogen content': safe_float(row.get('Hydrogen (% m/m)')),
        'Oxygen content': safe_float(row.get('Oxygen (% m/m)')),
        'Nitrogen content': safe_float(row.get('Nitrogen (% m/m)')),
        'Sulphur content': safe_float(row.get('Sulphur (% m/m)')),
    }

    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in stats.values()):
        return None

    return stats

def get_concawe_range(stats_df, cas):
    row = stats_df.loc[stats_df['CAS'] == cas]

    if row.empty:
        return None  # CAS not found → drop later

    row = row.iloc[0]

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    stats = {
        'Min carbon': safe_float(row.get('Cmin')),
        'Max carbon': safe_float(row.get('Cmax')),
    }

    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in stats.values()):
        return None

    return stats


def get_pca_stats(cas):
    supp_1 = pd.read_excel("House_supp_1.xlsx")
    supp_4 = pd.read_excel("House_supp_4.xlsx")

    s1_row = supp_1.loc[supp_1['CAS Number'] == cas]
    if s1_row.empty:
        return {}

    sample_id = s1_row.iloc[0]['Sample.ID']

    s4_row = supp_4.loc[supp_4['Sample.ID'] == sample_id]
    if s4_row.empty:
        return {}
    s4_row = s4_row.iloc[0]

    def safe_float(v):
        try:
            return float(v)
        except:
            return None

    return {
        'total.Wt.Percent': safe_float(s4_row.get('total.Wt.Percent')),
        'P3_7_index': safe_float(s4_row.get('P3_7_index')),
        'P4_7_index': safe_float(s4_row.get('P4_7_index')),
        'P5_7_index': safe_float(s4_row.get('P5_7_index')),
        'P1_2_index': safe_float(s4_row.get('P1_2_index')),
        'P1_index': safe_float(s4_row.get('P1_index')),
        'P2_index': safe_float(s4_row.get('P2_index')),
        'P3_index': safe_float(s4_row.get('P3_index')),
        'P4_index': safe_float(s4_row.get('P4_index')),
        'P5_index': safe_float(s4_row.get('P5_index')),
        'P6_index': safe_float(s4_row.get('P6_index')),
        'P7_index': safe_float(s4_row.get('P7_index')),
        'Primary_PAH': safe_float(s4_row.get('Primary_PAH')),
    }


def compute_carbon_heteroatom(df):
    min_c = int(df['Total Carbons'].min())
    max_c = int(df['Total Carbons'].max())

    stats = {
        'Min carbon': min_c,
        'Max carbon': max_c
    }

    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in stats.values()):
        return None

    return stats

import pandas as pd

def tally_category_info(df,
                        category_col="Class",
                        description_col="Description",
                        cmax_col="Cmax",
                        carbon_col="Carbon (% m/m)",
                        manual_classifier_keywords=("Manual", "Classifiers")):

    # Identify manual classifier columns by keyword
    manual_cols = [
        col for col in df.columns
        if any(kw.lower() in col.lower() for kw in manual_classifier_keywords)
    ]

    # Group and tally
    tally = (
        df.groupby(category_col)
          .apply(lambda g: pd.Series({
            #  "n_substances": len(g),
              "n_with_description": g[description_col].notna().sum(),
              "n_with_Cmax": g[cmax_col].notna().sum(),
              "n_with_carbon_%_m_m": g[carbon_col].notna().sum(),
              "n_with_manual_classifier": g[manual_cols].notna().any(axis=1).sum()
          }))
          .reset_index()
    )

    return tally


if __name__ == "__main__":
    cas_df = pd.read_excel("House_data_compiled.xlsx")
    manual_classifiers = cas_df["Manual classifiers"]
    all_fp_classifiers_df = pd.read_excel("Petroleum_classifiers.xlsx")
    all_fp_classifiers = all_fp_classifiers_df["Name"].dropna().tolist()
    cas_numbers = cas_df.iloc[:, 0].dropna().astype(str).tolist()

    df = pd.read_excel("Manual_classification_data.xlsx", sheet_name="Output+descriptions")
    result = tally_category_info(df)
    print(result)

    # print("\n --------------------------------------------------")
    # df_1_1, missing_1_1, smiles_c, man_empty = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["none"],
    #     classifiers_from=["smiles"]
    # )
    # df_1_1.to_csv("1_1_smiles_fps.csv", index=False)
    #
    # print("\n 2.1 Manual only")
    # df_2_1, missing_2_1, smiles_empty, manual_c = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["none"],
    #     classifiers_from=["manual"]
    # )
    # df_2_1.to_csv("2_1_manual_fps.csv", index=False)
    #
    # print(smiles_c)
    # print(len(smiles_c))
    # print(manual_c)
    # print(len(manual_c))
    # shared_c = []
    # for classifier in smiles_c:
    #     if classifier in manual_c:
    #         shared_c.append(classifier)
    # print(shared_c)
    # print(len(shared_c))
    # exit()

    # ----------- SMILES ------------
    print("\n --------------------------------------------------")
    print("\n FINGERPRINTS FROM SMILES-BASED CLASSIFIERS")
    print("\n 1.1 SMILES only")
    df_1_1, missing_1_1 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["none"],
        classifiers_from=["smiles"]
    )
    df_1_1.to_csv("1_1_smiles_fps.csv", index=False)

    print("\n 1.2 SMILES with calculated range")
    df_1_2, missing_1_2 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["chem"],
        classifiers_from=["smiles"]
    )
    df_1_2.to_csv("1_2_smiles_range_fps.csv", index=False)

    print("\n 1.3 SMILES with PAH content")
    df_1_3, missing_1_3 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah"],
        classifiers_from=["smiles"]
    )
    df_1_3.to_csv("1_3_smiles_pah_fps.csv", index=False)

    print("\n 1.4 SMILES with calculated range and PAH content")
    df_1_4, missing_1_4 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["chem", "pah"],
        classifiers_from=["smiles"]
    )
    df_1_4.to_csv("1_4_smiles_range_pah_fps.csv", index=False)

    print("\n 1.5 Calculated range only")
    df_1_5, missing_1_5 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["chem"],
        classifiers_from=["none"]
    )
    df_1_5.to_csv("1_5_calculated_range_fps.csv", index=False)

    print("\n 1.6 SMILES and composition")
    df_1_6, missing_1_6 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp"],
        classifiers_from=["smiles"]
    )
    df_1_6.to_csv("1_6_smiles_comp_fps.csv", index=False)

    print("\n 1.7 SMILES, composition, and range")
    df_1_7, missing_1_7 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["chem", "concawe_comp"],
        classifiers_from=["smiles"]
    )
    df_1_7.to_csv("1_7_smiles_comp_range_fps.csv", index=False)

    print("\n 1.8 SMILES, composition and pah")
    df_1_8, missing_1_8 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_comp"],
        classifiers_from=["smiles"]
    )
    df_1_8.to_csv("1_8_smiles_comp_pah_fps.csv", index=False)

    print("\n 1.9 SMILES, composition, pah, and range")
    df_1_9, missing_1_9 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["chem", "concawe_comp", "pah"],
        classifiers_from=["smiles"]
    )
    df_1_9.to_csv("1_9_smiles_comp_range_pah_fps.csv", index=False)

    # ----------- MANUAL ------------
    print("\n --------------------------------------------------")
    print("\n FINGERPRINTS FROM MANUAL CLASSIFIERS")
    print("\n 2.1 Manual only")
    df_2_1, missing_2_1 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["none"],
        classifiers_from=["manual"]
    )
    df_2_1.to_csv("2_1_manual_fps.csv", index=False)


    print("\n 2.2 Manual with range from Concawe")
    df_2_2, missing_2_2 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_range"],
        classifiers_from=["manual"]
    )
    df_2_2.to_csv("2_2_manual_range_fps.csv", index=False)

    print("\n 2.3 Manual with PAH content")
    df_2_3, missing_2_3 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah"],
        classifiers_from=["manual"]
    )
    df_2_3.to_csv("2_3_manual_pah_fps.csv", index=False)

    print("\n 2.4 Manual with range and PAH content")
    df_2_4, missing_2_4 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah","concawe_range"],
        classifiers_from=["manual"]
    )
    df_2_4.to_csv("2_4_manual_pah_range_fps.csv", index=False)

    print("\n 2.5 Manual with composition")
    df_2_5, missing_2_5 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp"],
        classifiers_from=["manual"]
    )
    df_2_5.to_csv("2_5_manual_comp_fps.csv", index=False)

    print("\n 2.6 Manual with composition, pah")
    df_2_6, missing_2_6 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_comp"],
        classifiers_from=["manual"]
    )
    df_2_6.to_csv("2_6_manual_pah_comp_fps.csv", index=False)

    print("\n 2.7 Manual with composition, range")
    df_2_7, missing_2_7 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_range", "concawe_comp"],
        classifiers_from=["manual"]
    )
    df_2_7.to_csv("2_7_manual_comp_range_fps.csv", index=False)

    print("\n 2.8 Manual with composition, range, pah")
    df_2_8, missing_2_8 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_range", "concawe_comp"],
        classifiers_from=["manual"]
    )
    df_2_8.to_csv("2_8_manual_comp_pah_range_fps.csv", index=False)

    # ----------- COMPOSITIONAL ------------
    print("\n --------------------------------------------------")
    print("\n FINGERPRINTS FROM COMPOSITION DATA")
    print("\n 3.1 Range") # make sure to compare with with
    df_3_1, missing_3_1 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_range"],
        classifiers_from=["none"]
    )
    df_3_1.to_csv("3_1_description_range_fps.csv", index=False)

    print("\n 3.2 PAH")
    df_3_2, missing_3_2 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah"],
        classifiers_from=["none"]
    )
    df_3_2.to_csv("3_2_pah_fps.csv", index=False)

    print("\n 3.3 % Composition")
    df_3_3, missing_3_3 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp"],
        classifiers_from=["none"]
    )
    df_3_3.to_csv("3_3_comp_fps.csv", index=False)

    print("\n 3.4 Range and PAH")
    df_3_4, missing_3_4 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_range"],
        classifiers_from=["none"]
    )
    df_3_4.to_csv("3_4_range_pah_fps.csv", index=False)

    print("\n 3.5 Range and % Composition")
    df_3_5, missing_3_5 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_range", "concawe_comp"],
        classifiers_from=["none"]
    )
    df_3_5.to_csv("3_5_range_comp_fps.csv", index=False)

    print("\n 3.6 PAH and % Composition")
    df_3_6, missing_3_6 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_comp"],
        classifiers_from=["none"]
    )
    df_3_6.to_csv("3_6_pah_composition_fps.csv", index=False)

    print("\n 3.7 PAH and range and % Composition")
    df_3_7, missing_3_7 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_range", "concawe_comp"],
        classifiers_from=["none"]
    )
    df_3_7.to_csv("3_7_pah_range_comp_fps.csv", index=False)

    # ------------ BOTH ---------------------
    print("\n 4.1 Manual with SMILES")
    df_4_1, missing_4_1 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["none"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_1.to_csv("4_1_manual_smiles_fps.csv", index=False)

    print("\n 4.2 Manual with SMILES and PAH")
    df_4_2, missing_4_2 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_2.to_csv("4_2_manual_smiles_pah_fps.csv", index=False)

    print("\n 4.3 Manual with SMILES and range")
    df_4_3, missing_4_3 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_range"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_3.to_csv("4_3_manual_smiles_range_fps.csv", index=False)

    print("\n 4.4 Manual with SMILES and comp")
    df_4_4, missing_4_4 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_4.to_csv("4_4_manual_smiles_comp_fps.csv", index=False)

    print("\n 4.5 Manual with SMILES, comp, range")
    df_4_5, missing_4_5 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp", "concawe_range"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_5.to_csv("4_5_manual_smiles_range_comp_fps.csv", index=False)

    print("\n 4.6 Manual with SMILES, pah, range")
    df_4_6, missing_4_6 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["pah", "concawe_range"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_6.to_csv("4_6_manual_smiles_pah_range_fps.csv", index=False)

    print("\n 4.7 Manual with SMILES, pah, comp")
    df_4_7, missing_4_7 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp", "pah"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_7.to_csv("4_7_manual_smiles_comp_pah_fps.csv", index=False)

    print("\n 4.8 Manual with SMILES, pah, comp")
    df_4_8, missing_4_8 = summarize_cas_files(
        cas_numbers,
        all_fp_classifiers,
        manual_classifiers,
        cas_df,
        stats_to_include=["concawe_comp", "pah", "concawe_range"],
        classifiers_from=["manual", "smiles"]
    )
    df_4_8.to_csv("4_8_manual_smiles_comp_pah_range_fps.csv", index=False)





    # # Manual-only output
    # print("Fingerprints from manual classifications")
    # df_manual, missing_manual = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["concawe_range"],
    #     classifiers_from=["manual"]
    # )
    # df_manual.to_csv("Manual_only_feature_vectors.csv", index=False)
    #
    # # SMILES-only output
    # print("\nFingerprints from SMILES classifications")
    # df_smiles, missing_smiles = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["chem"],
    #     classifiers_from=["smiles"]
    # )
    # df_smiles.to_csv("SMILES_only_feature_vectors.csv", index=False)
    #
    # # Manaul + SMILES output
    # print('\nFingerprints from manual + SMILES classifications')
    # df_both, missing_both = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["concawe_range", "chem"],
    #     classifiers_from=["smiles", "manual"]
    # )
    # df_both.to_csv("Both_feature_vectors.csv", index=False)
    #
    # # Only data
    # print('\nFingerprints data classifications')
    # df_both, missing_both = summarize_cas_files(
    #     cas_numbers,
    #     all_fp_classifiers,
    #     manual_classifiers,
    #     cas_df,
    #     stats_to_include=["concawe_comp", "concawe_range", "pah"],
    #     classifiers_from=["none"]
    # )
    # df_both.to_csv("Measured_feature_vectors.csv", index=False)


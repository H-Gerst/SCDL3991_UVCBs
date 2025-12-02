import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, fowlkes_mallows_score

def class_shared_tally(clustering_a, clustering_b, category_col, output_folder):
    clean_a = clustering_a[['CAS']]
    clean_b = clustering_b[['CAS', category_col]]

    common_cas = set(clean_a['CAS']).intersection(clean_b['CAS'])

    shared_rows = clean_b[clean_b['CAS'].isin(common_cas)]

    tally = shared_rows[category_col].value_counts().sort_index()

    tally_df = tally.reset_index()
    tally_df.columns = [category_col, 'Shared CAS Count']

    tally_df.to_csv(os.path.join(output_folder, f"{category_col}_shared_cas.csv"), index=False)

    return tally_df

def find_common_CAS(clustering_a, clustering_b, category_col):
    clean_cluster_a = clustering_a[['CAS', 'Cluster']]
    clean_cluster_b = clustering_b[['CAS', category_col]]

    print(len(set(clean_cluster_a['Cluster'])))
    print(len(set(clean_cluster_b[category_col].dropna())))

    # lookup dictionary for the variable category column
    class_lookup = dict(zip(clean_cluster_b['CAS'], clean_cluster_b[category_col]))

    rows = []
    for _, row in clean_cluster_a.iterrows():
        cas = row['CAS']
        cluster = row['Cluster']

        if cas not in class_lookup:
            continue

        house_category = class_lookup[cas]

        rows.append({
            'CAS': cas,
            'Cluster': cluster,
            category_col: house_category
        })

    common_df = pd.DataFrame(rows)

    print(f"Unique clusters: {len(common_df['Cluster'].unique())}")
    print(f"Unique {category_col} groups: {len(common_df[category_col].unique())}")

    return common_df

def qualitative_compare(common_df, category_col, output_folder, cluster_labels=None):
    """
    Plot a heatmap comparing clusters vs categories with colors matching cluster palette.
    Y-axis shows cluster numbers, X-axis shows category names.
    Each cluster row is colored to match the cluster palette.
    """
    # Create contingency table
    contingency = pd.crosstab(common_df['Cluster'], common_df[category_col])
   # print(contingency)
    contingency.to_csv(os.path.join(output_folder, f"{category_col}_contingency.csv"), index=False)

    if category_col == "Category":
        x_lab = "Bioactivity group"
    else:
        x_lab = "Manufacturing class"

    plt.figure(figsize=(24, 28))

    # Define cluster palette
    if cluster_labels is None:
        cluster_labels = sorted(contingency.index.unique())
    n_clusters = len(cluster_labels)
    palette = sns.color_palette("tab20", n_colors=min(n_clusters, 20))
    if n_clusters > 20:
        palette += sns.color_palette("husl", n_colors=n_clusters - 20)
    row_colors = [palette[i % len(palette)] for i in range(n_clusters)]

    # Normalize heatmap data
    heat_data = contingency.values.astype(float)
    max_val = heat_data.max()
    colored_data = np.zeros_like(heat_data, dtype=float)
    for i in range(n_clusters):
        if max_val > 0:
            colored_data[i, :] = heat_data[i, :] / max_val
        else:
            colored_data[i, :] = 0

    # Plot heatmap
    ax = sns.heatmap(
        colored_data,
        annot=contingency.values,
        fmt='d',
        annot_kws={"size": 35},
        linewidths=0.5,
        linecolor='white',
        cbar=False,
        cmap='Blues'  # base cmap for intensity
    )

    # Apply row colors to tick labels
    ax.set_yticklabels(contingency.index, fontsize=35, rotation=0)
    ax.set_xticklabels(contingency.columns, fontsize=35, rotation=45, ha='right')

    # Color the rows in the heatmap by cluster
    for ytick, color in zip(ax.get_yticklabels(), row_colors):
        ytick.set_color(color)

    # Axis labels
    ax.set_xlabel(f'{x_lab}', fontsize=40, labelpad=15)
    ax.set_ylabel("Cluster", fontsize=40, labelpad=20)

    # Remove gridlines
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{category_col}_Heatmap.png"), dpi=300)
    # plt.show()

def quantitative_compare(common_df, category_col):
    """
    Quantitatively compare two clusterings using ARI, NMI, and V-measure.
    Returns a dictionary of scores.
    """
    # Encode clusters as integers
    le_cluster = LabelEncoder()
    le_category = LabelEncoder()

    cluster_labels = le_cluster.fit_transform(common_df['Cluster'])
    category_labels = le_category.fit_transform(common_df[category_col])

    ari = adjusted_rand_score(category_labels, cluster_labels)
    nmi = normalized_mutual_info_score(category_labels, cluster_labels)
    vmeasure = v_measure_score(category_labels, cluster_labels)
    fmi = fowlkes_mallows_score(category_labels, cluster_labels)

    scores = {
        "Adjusted Rand Index": ari,
        "Normalized Mutual Information": nmi,
        "V-measure": vmeasure,
        "Fowlkes mallows score": fmi
    }

    print("\nQuantitative comparison scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    return scores

def make_it_neat(fp_list,house_clusters,category_col):
    print(f'Clustering comparison for {category_col}')
    for fp in fp_list:
        print(f'\n--- {fp} ---')
        output_file = f'{fp}_cluster_results'
        hierarchical_clusters = pd.read_csv(f'{fp}_cluster_results/Hierarchical_Cluster_Members.csv')
        kmeans_clusters = pd.read_csv(f'{fp}_cluster_results/Kmeans_Cluster_Members.csv')

        if not hierarchical_clusters.empty:
            print("Hierarchical cluster comparison")
            h_alignment = find_common_CAS(hierarchical_clusters, house_clusters, category_col)
            class_shared_tally(hierarchical_clusters, house_clusters, category_col, output_file)
            qualitative_compare(h_alignment, category_col, output_file)
            quantitative_compare(h_alignment, category_col)

        if not kmeans_clusters.empty:
            print("\nKmeans cluster comparison")
            k_alignment = find_common_CAS(kmeans_clusters, house_clusters, category_col)
            class_shared_tally(kmeans_clusters, house_clusters, category_col, output_file)
            qualitative_compare(k_alignment, category_col, output_file)
            quantitative_compare(k_alignment, category_col)


if __name__ == "__main__":
    house_clusters = pd.read_excel("House_data_compiled.xlsx")
    fp_df = pd.read_excel("Fingerprint_list.xlsx")
    fp_list = fp_df.iloc[:, 0].dropna().astype(str).tolist()
    fp_smiles = fp_list[:9]
    fp_manual = fp_list[9:17]
    fp_compositional = fp_list[17:24]
    fp_both = fp_list[24:]

    # ----------- SMILES ------------
    print("\n --------------------------------------------------")
    print("Comparison of CLUSTERING FROM SMILES-BASED CLASSIFIERS")
    make_it_neat(fp_smiles, house_clusters, "Category")
    make_it_neat(fp_smiles, house_clusters, "Class (16)")
    #
    # ----------- Manual ------------
    print("\n --------------------------------------------------")
    print("comparison of CLUSTERING FROM MANUAL CLASSIFIERS")
    make_it_neat(fp_manual,house_clusters, "Category")
    make_it_neat(fp_manual, house_clusters, "Class (16)")

    # ----------- Compositional ------------
    print("\n --------------------------------------------------")
    print("comparison of CLUSTERING FROM CHARACTERISATION DATA")
    make_it_neat(fp_compositional,house_clusters, "Category")
    make_it_neat(fp_compositional, house_clusters, "Class (16)")

    # ---------- Both -----------------
    print('\n ----------------------------------------------------')
    print("comparison of CLUSTERING FROM SMILES AND MANUAL")
    make_it_neat(fp_both, house_clusters, 'Category')
    make_it_neat(fp_both, house_clusters, 'Class (16)')

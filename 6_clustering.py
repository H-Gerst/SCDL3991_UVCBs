import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.exceptions import ConvergenceWarning
from adjustText import adjust_text


def align_cas(feature_vectors, cas_df):
    feature_vectors['CAS'] = feature_vectors['CAS'].astype(str).str.strip()
    cas_df['CAS'] = cas_df['CAS'].astype(str).str.strip()

    cas_feat = set(feature_vectors['CAS'])
    cas_house = set(cas_df['CAS'])

    shared = cas_feat & cas_house

    # Restrict to only shared CAS
    fv_aligned = feature_vectors[feature_vectors['CAS'].isin(shared)].copy()
    cas_aligned = cas_df[cas_df['CAS'].isin(shared)].copy()

    # Sort by CAS to guarantee identical ordering
    fv_aligned = fv_aligned.sort_values("CAS").reset_index(drop=True)
    cas_aligned = cas_aligned.sort_values("CAS").reset_index(drop=True)

    return fv_aligned, cas_aligned


# ======================================================
# 1. HIERARCHICAL CLUSTERING
# ======================================================
def run_hierarchical_clustering(data, cluster_range=range(2, 11), method='average', metric='euclidean'):
    Z = linkage(data, method=method, metric=metric)
    score_dict = {}
    best_score = -1
    best_k = None
    best_labels = None

    for k in cluster_range:
        labels = fcluster(Z, k, criterion='maxclust') - 1  # zero-based
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(data, labels)
        score_dict[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    return best_labels, best_score, best_k, score_dict, Z

# ======================================================
# 2. KMEANS CLUSTERING
# ======================================================
def run_kmeans_clustering(data, cluster_range=range(2, 11), random_state=42):
    best_score = -1
    best_k = None
    best_model = None
    score_dict = {}
    for k in cluster_range:
        if k >= len(data):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            labels = model.fit_predict(data)

        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(data, labels)
        score_dict[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    labels = best_model.labels_ if best_model is not None else np.zeros(len(data), dtype=int)
    return labels, best_score, best_k, score_dict, best_model

# ======================================================
# 3. FEATURE IMPORTANCE (raw features)
# ======================================================
def feature_importance_by_cluster(data, labels, features=None):
    df = pd.DataFrame(data, columns=features if features is not None else [f'Feature_{i}' for i in range(data.shape[1])])
    df['Cluster'] = labels
    cluster_means = df.groupby('Cluster').mean()
    feature_variances = cluster_means.var().sort_values(ascending=False)
    return pd.DataFrame({'Feature': feature_variances.index, 'Variance_across_clusters': feature_variances.values}).reset_index(drop=True)

# ======================================================
# 4. PCA-based Cluster Separation
# ======================================================
def pca_cluster_separation(data, labels, n_components=10):
    n_components = min(n_components, data.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(data)
    df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_components)])
    df['Cluster'] = labels
    separation_scores = []
    for pc in df.columns[:-1]:
        cluster_means = df.groupby('Cluster')[pc].mean()
        between_var = cluster_means.var()
        within_var = df.groupby('Cluster')[pc].var().mean()
        score = between_var / (within_var + 1e-12)
        separation_scores.append({'PC': pc, 'BetweenVar': between_var, 'WithinVar': within_var, 'SeparationScore': score})
    return pd.DataFrame(separation_scores).sort_values('SeparationScore', ascending=False)

# ======================================================
# 5. TIDY CLUSTER MEMBERSHIP TABLE
# ======================================================
def summarize_descriptions_by_cluster(
    cas_df,
    labels,
    cas_col='CAS',
    description_col='Description',
    class_col='Class (16)',
    save_path=None
):
    # Handle mismatched lengths safely
    if len(cas_df) != len(labels):
        min_len = min(len(cas_df), len(labels))
        cas_df = cas_df.iloc[:min_len, :].copy()
        labels = labels[:min_len]

    cas_df = cas_df.copy()
    cas_df['Cluster'] = labels

    # Only keep available columns (avoid errors if missing class_col)
    cols_to_keep = ['Cluster', cas_col, description_col]
    if class_col in cas_df.columns:
        cols_to_keep.append(class_col)

    cluster_summary = cas_df[cols_to_keep].copy()

    if save_path:
        cluster_summary.to_csv(save_path, index=False)

    return cluster_summary


# ======================================================
# 6. VISUALIZATION & EXPORT
# ======================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from adjustText import adjust_text

# # def visualize_and_export(data, labels, method_name, out_dir, features=None, cas_df=None,
#
# def visualize_and_export(data, labels, method_name, out_dir, features=None, cas_df=None, score=None, Z=None, n_pca_components=10):
#     sns.set_style("whitegrid")
#     os.makedirs(out_dir, exist_ok=True)
#
#     n_clusters = len(np.unique(labels))
#     palette = sns.color_palette("tab20", n_colors=n_clusters)
#
#     df_features = pd.DataFrame(data, columns=features)
#     df_numeric = df_features.select_dtypes(include=[np.number])
#
#     # ---------------------------
#     # PCA
#     # ---------------------------
#     pca = PCA(n_components=2)
#     components = pca.fit_transform(data)
#
#     loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
#     loading_strength = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
#     top_idx = np.argsort(loading_strength)[-10:]  # top 10 features
#
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', '*', 'X']
#     base_sizes = [120, 140, 160, 180]  # will scale up for plot
#
#     fig, ax = plt.subplots(figsize=(30, 18))
#     plt.rcParams.update({'font.size': 40})
#
#     # ---------------------------
#     # Scatter points
#     # ---------------------------
#     for i, cluster_id in enumerate(np.unique(labels)):
#         cluster_points = components[labels == cluster_id]
#         ax.scatter(
#             cluster_points[:, 0], cluster_points[:, 1],
#             color=palette[i % len(palette)],
#             s=base_sizes[i % len(base_sizes)] * 2,  # make points larger
#             marker=markers[i % len(markers)],
#             edgecolor='k',
#             alpha=0.85,
#             label=f'Cluster {cluster_id}'
#         )
#
#     # ---------------------------
#     # Legend
#     # ---------------------------
#     title_str = f"{method_name} PCA Projection"
#     if score is not None:
#         title_str += f" (Silhouette: {score:.3f})"
#
#     ax.legend(
#         bbox_to_anchor=(1.05, 1),
#         loc='upper left',
#         title='Clusters',
#         fontsize=30,
#         title_fontsize=30,
#         ncol=2,
#         frameon=False,
#         markerscale=2.0,  # enlarge legend markers
#         handlelength=3
#     )
#
#     # ---------------------------
#     # Adjust text for PCA vectors
#     # ---------------------------
#     texts = []
#     numeric_columns = df_numeric.columns.tolist()
#     scale = 2.0
#
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     for i in top_idx:
#         feature = numeric_columns[i]
#         x, y = loadings[i, 0] * scale, loadings[i, 1] * scale
#         vector_length = np.sqrt(loadings[i, 0] ** 2 + loadings[i, 1] ** 2)
#
#         # Offset based on quadrant
#         offset_x = 15 if x >= 0 else -15
#         offset_y = 15 if y >= 0 else -15
#
#         # Make sure labels don't go beyond plot area
#         if x + offset_x / 10 > xlim[1] * 0.85:
#             offset_x = -offset_x
#
#         t = ax.annotate(
#             f"{feature} ({vector_length:.2f})",
#             xy=(x, y),
#             xytext=(offset_x, offset_y),
#             textcoords="offset points",
#             fontsize=32,
#             fontweight='bold',
#             ha='center',
#             va='center',
#             arrowprops=dict(
#                 arrowstyle="->",
#                 lw=1,
#                 color='grey',
#                 shrinkA=10,  # shorter arrows
#                 shrinkB=5
#             )
#         )
#         texts.append(t)
#
#     adjust_text(
#         texts,
#         ax=ax,
#         only_move={'points': 'xy', 'text': 'xy'},
#         force_points=3.0,
#         force_text=3.0,
#         expand_text=(2.0, 2.0),
#         expand_points=(3.0, 3.0),
#         arrowprops=dict(
#             arrowstyle="->",
#             lw=1,
#             color='grey',
#             shrinkA=10,
#             shrinkB=5
#         ),
#         lim=3000,
#         autoalign=False
#     )
#
#     # ---------------------------
#     # Axis labels with variance
#     # ---------------------------
#     ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=40, labelpad=15)
#     ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=40, labelpad=15)
#
#     # ---------------------------
#     # Axes appearance
#     # ---------------------------
#     for spine in ax.spines.values():
#         spine.set_linewidth(2.5)
#         spine.set_color('black')
#
#     ax.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False,
#                    length=12, width=2, direction='out', labelsize=32)
#
#     # ---------------------------
#     # Prevent labels going into legend
#     # ---------------------------
#     ax.set_xlim(xlim[0], xlim[1] * 0.85)  # leave space on right for legend
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"{method_name}_PCA.png"), dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()


def visualize_and_export(data, labels, method_name, out_dir, features=None, cas_df=None, score=None, Z=None, n_pca_components=10):
    sns.set_style("whitegrid")
    os.makedirs(out_dir, exist_ok=True)
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("tab20", n_colors=n_clusters)

    df_features = pd.DataFrame(data, columns=features)
    df_numeric = df_features.select_dtypes(include=[np.number])

    # ---------------------------
    # PCA
    # ---------------------------
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_strength = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_idx = np.argsort(loading_strength)[-10:]  # top_n_features=10

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', '*', 'X']
    base_sizes = [120, 140, 160, 180]

    fig, ax = plt.subplots(figsize=(30, 18))
    plt.rcParams.update({'font.size': 40})

    # ---------------------------
    # Scatter points
    # ---------------------------
    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_points = components[labels == cluster_id]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            color=palette[i % len(palette)],
            s=base_sizes[i % len(base_sizes)]*6,
            marker=markers[i % len(markers)],
            edgecolor='k',
            alpha=0.85,
            label=f'Cluster {cluster_id}'
        )

    # Legend
    title_str = f"{method_name} PCA Projection"
    if score is not None:
        title_str += f" (Silhouette: {score:.3f})"

   # ax.set_title(title_str, fontsize=20)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=28,
        ncol=2,
        markerscale=0.8,
        frameon=False
    )

    # ---------------------------
    # Adjust text for PCA vectors
    # ---------------------------
    texts = []
    numeric_columns = df_numeric.columns.tolist()
    scale = 2.0

    for i in top_idx:
        feature = numeric_columns[i]
        x, y = loadings[i, 0] * scale, loadings[i, 1] * scale
        vector_length = np.sqrt(loadings[i, 0] ** 2 + loadings[i, 1] ** 2)

        # Offset based on quadrant
        offset_x = 15 if x >= 0 else -15
        offset_y = 15 if y >= 0 else -15

        t = ax.annotate(
            f"{feature} ({vector_length:.2f})",
            xy=(x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=30,
            fontweight='bold',
            ha='center',
            va='center',
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color='grey',
                shrinkA=12,  # larger shrink to avoid striking text
                shrinkB=6
            )
        )
        texts.append(t)

    adjust_text(
        texts,
        only_move={'points': 'xy', 'text': 'xy'},
        force_points=9.0,
        force_text=9.0,
        expand_text=(15.0, 15.0),
        expand_points=(12.0, 12.0),
        # arrowprops=dict(
        #     arrowstyle="->",
        #     lw=1,
        #     color='grey',  # grey arrows
        #     shrinkA=10,
        #     shrinkB=5
        # ),
        lim=3000,
        autoalign = False,  # keeps labels from rotating unexpectedly
        ax=ax,
    )

    # ---------------------------
    # Thicker black plot border
    # ---------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # ---------------------------
    # Re-enable tick marks on axes
    # ---------------------------
    ax.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False,
                   length=15, width=2, direction='out', labelsize=40)

    # Axis labels with PCA number and % variance
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
                  fontsize=40, labelpad=20)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
                  fontsize=40, labelpad=20)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{method_name}_PCA.png"), dpi=300, bbox_inches='tight')

    # Save PCA
    plt.savefig(os.path.join(out_dir, f"{method_name}_PCA.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    # Bar plot: CAS per cluster


    cluster_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=palette)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    for spine in ['left', 'bottom']:
        plt.gca().spines[spine].set_linewidth(2.5)
    ax.tick_params(axis='x', direction='out', length=18, width=3, labelsize=50)
    ax.tick_params(axis='y', direction='out', length=18, width=3, labelsize=50)
    plt.title(f'{method_name} - Number of CAS per Cluster', fontsize=20)
    plt.xlabel('Cluster', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{method_name}_Cluster_Counts.png"), dpi=300)
    plt.close()

    # Membership table
    if cas_df is not None:
        membership_summary = summarize_descriptions_by_cluster(
            cas_df, labels,
            cas_col='CAS',
            description_col='Description',
            class_col='Class (16)',
            save_path=os.path.join(out_dir, f"{method_name}_Cluster_Members.csv")
        )
    else:
        membership_summary = pd.DataFrame({'Cluster': labels})

    # Feature importance
    feature_importance = feature_importance_by_cluster(data, labels, features)
    feature_importance.to_csv(os.path.join(out_dir, f"{method_name}_Feature_Importance.csv"), index=False)

    # PCA cluster separation
    separation_df = pca_cluster_separation(data, labels, n_components=n_pca_components)
    separation_df.to_csv(os.path.join(out_dir, f"{method_name}_PCA_Cluster_Separation.csv"), index=False)

    # Dendrogram
    if method_name.lower().startswith("hierarchical") and Z is not None:
        plt.figure(figsize=(18,8))
        dendrogram(Z, truncate_mode='lastp', p=30,
                   leaf_rotation=90., leaf_font_size=10.,
                   show_contracted=True)
        plt.title("Hierarchical Clustering Dendrogram", fontsize=20)
        plt.xlabel("Cluster Size", fontsize=16)
        plt.ylabel("Distance", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{method_name}_Dendrogram.png"), dpi=300)
        plt.close()

    # Summary
    with open(os.path.join(out_dir, f"{method_name}_Summary.txt"), 'w') as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        if score is not None:
            f.write(f"Silhouette Score: {score:.3f}\n")

    return membership_summary, feature_importance, separation_df


# ======================================================
# 7. MAIN PIPELINE
# ======================================================
def main_clustering_pipeline(X, cas_df, feature_names, cluster_range=range(2,11), output_folder="Cluster_Results"):
    os.makedirs(output_folder, exist_ok=True)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Hierarchical ---
    print("Hierarchical Clustering")
    hier_labels, hier_score, hier_best_k, hier_scores, Z = run_hierarchical_clustering(X_scaled, cluster_range)
    print(f"Best k = {hier_best_k} (Silhouette: {hier_score:.3f})")
    plt.figure(figsize=(10,6))
    plt.plot(list(hier_scores.keys()), list(hier_scores.values()), marker='o')
    plt.title("Hierarchical Silhouette Scores", fontsize=18)
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Silhouette Score", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Hierarchical_Silhouette_Scores.png"), dpi=300)
    plt.close()

    visualize_and_export(X_scaled, hier_labels, "Hierarchical", output_folder, features=feature_names, cas_df=cas_df, score=hier_score, Z=Z)

    # --- KMeans ---
    print("\nKMeans Clustering")
    kmeans_labels, kmeans_score, kmeans_best_k, kmeans_scores, kmeans_model = run_kmeans_clustering(X_scaled, cluster_range)
    print(f"Best k = {kmeans_best_k} (Silhouette: {kmeans_score:.3f})")
    plt.figure(figsize=(10,6))
    plt.plot(list(kmeans_scores.keys()), list(kmeans_scores.values()), marker='o')
    plt.title("KMeans Silhouette Scores", fontsize=18)
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Silhouette Score", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "KMeans_Silhouette_Scores.png"), dpi=300)
    plt.close()

    visualize_and_export(X_scaled, kmeans_labels, "KMeans", output_folder, features=feature_names, cas_df=cas_df, score=kmeans_score)

    print(f"\n All results exported to: {output_folder}")

def check_for_mismatch(my_cas,house_cas):
    # Extract CAS sets
    cas_feat = set(my_cas['CAS'])
    cas_cluster = set(house_cas['CAS'])

    # Compute matches and mismatches
    shared = cas_feat.intersection(cas_cluster)
    only_feat = cas_feat - cas_cluster
    only_cluster = cas_cluster - cas_feat

    print("--- CAS Comparison --")
    print(f"Total in feature vectors: {len(cas_feat)}")
    print(f"Total in cluster members: {len(cas_cluster)}")
    print(f"Shared CAS numbers: {len(shared)}")
    print(f"Missing from cluster results: {len(only_feat)}")
    print(f"Extra in cluster results: {len(only_cluster)}")

    if only_feat:
        print("\n-- CAS present in feature_vectors.csv but NOT in cluster results --")
        print(sorted(list(only_feat)))

    if only_cluster:
        print("\n-- CAS present in cluster results but NOT in feature_vectors.csv --")
        print(sorted(list(only_cluster)))

def the_better_way(fp_list,cas_df):
    for fp in fp_list:
        print(f'\n --- {fp} ---')
        fp_file = f"{fp}.csv"
        if os.path.exists(fp_file):
            feature_vectors = pd.read_csv(fp_file)
            if not feature_vectors.empty:
                fv_aligned, cas_aligned = align_cas(feature_vectors, cas_df.copy())

                X = fv_aligned.iloc[:, 1:].values
                feature_names = fv_aligned.columns[1:].tolist()

                main_clustering_pipeline(
                    X=X,
                    cas_df=cas_aligned,
                    feature_names=feature_names,
                    cluster_range=range(3, 50),
                    output_folder=f"{fp}_cluster_results"
                )

                feat = feature_vectors
                cluster = pd.read_csv(f'{fp}_cluster_results/Hierarchical_Cluster_Members.csv')
                check_for_mismatch(feat, cluster)

            else:
                print(f'No fingerprints working here {fp}')
        else:
            print(f'Ha! no {fp} file')



if __name__ == "__main__":
    # Load the master CAS dataframe once
    cas_df = pd.read_excel("House_data_compiled.xlsx")
    fp_df = pd.read_excel("Fingerprint_list.xlsx")
    fp_list = fp_df.iloc[:, 0].dropna().astype(str).tolist()
    fp_smiles = fp_list[:9]
    fp_manual = fp_list[9:17]
    fp_compositional = fp_list[17:24]
    fp_both = fp_list[24:]


    # ----------- SMILES ------------
    print("\n --------------------------------------------------")
    print("CLUSTERING FROM SMILES-BASED CLASSIFIERS")
    #the_better_way(fp_smiles,cas_df)

    # ----------- Manual ------------
    print("\n --------------------------------------------------")
    print("CLUSTERING FROM MANUAL CLASSIFIERS")
    the_better_way(fp_manual,cas_df)

    # ----------- Compositional ------------
    print("\n --------------------------------------------------")
    print("CLUSTERING FROM CHARACTERISATION DATA")
    #the_better_way(fp_compositional,cas_df)

    # ----------- Both manual and SMILES
    print("\n --------------------------------------------------")
    print("BOTH CLASSSIFIERS")
    #the_better_way(fp_both, cas_df)



    # print("\n 1.1 SMILES only")
    #
    # fv_smiles = pd.read_csv("SMILES_only_feature_vectors.csv")
    # fv_aligned, cas_aligned = align_cas(fv_smiles, cas_df.copy())
    #
    # X = fv_aligned.iloc[:, 1:].values
    # feature_names = fv_aligned.columns[1:].tolist()
    #
    # main_clustering_pipeline(
    #     X=X,
    #     cas_df=cas_aligned,
    #     feature_names=feature_names,
    #     cluster_range=range(3, 50),
    #     output_folder="SMILES_only_cluster_results"
    # )
    #
    # feat = fv_smiles
    # cluster = pd.read_csv("SMILES_only_cluster_results/Hierarchical_Cluster_Members.csv")
    # check_for_mismatch(feat, cluster)
    #
    # # -----------------------------
    # # Manual only clustering
    # # -----------------------------
    # print("\n=== Manual only clustering ===")
    # fv_manual = pd.read_csv("Manual_only_feature_vectors.csv")
    # fv_aligned, cas_aligned = align_cas(fv_manual, cas_df.copy())
    #
    # X = fv_aligned.iloc[:, 1:].values
    # feature_names = fv_aligned.columns[1:].tolist()
    #
    # main_clustering_pipeline(
    #     X=X,
    #     cas_df=cas_aligned,
    #     feature_names=feature_names,
    #     cluster_range=range(3, 50),
    #     output_folder="Manual_only_cluster_results"
    # )
    #
    # feat = fv_manual
    # cluster = pd.read_csv("Manual_only_cluster_results/Hierarchical_Cluster_Members.csv")
    # check_for_mismatch(feat, cluster)
    #
    # # -----------------------------
    # # Both clustering
    # # -----------------------------
    # print("\n=== Both clustering ===")
    # fv_both = pd.read_csv("Both_feature_vectors.csv")
    # fv_aligned, cas_aligned = align_cas(fv_both, cas_df.copy())
    #
    # X = fv_aligned.iloc[:, 1:].values
    # feature_names = fv_aligned.columns[1:].tolist()
    #
    # main_clustering_pipeline(
    #     X=X,
    #     cas_df=cas_aligned,
    #     feature_names=feature_names,
    #     cluster_range=range(3, 50),
    #     output_folder="Both_cluster_results"
    # )
    #
    # feat = fv_both
    # cluster = pd.read_csv("Both_cluster_results/Hierarchical_Cluster_Members.csv")
    # check_for_mismatch(feat, cluster)
    #
    # # -----------------------------
    # # Measured clustering
    # # -----------------------------
    # print("\n=== Measured clustering ===")
    # fv_both = pd.read_csv("Measured_feature_vectors.csv")
    # fv_aligned, cas_aligned = align_cas(fv_both, cas_df.copy())
    #
    # X = fv_aligned.iloc[:, 1:].values
    # feature_names = fv_aligned.columns[1:].tolist()
    #
    # main_clustering_pipeline(
    #     X=X,
    #     cas_df=cas_aligned,
    #     feature_names=feature_names,
    #     cluster_range=range(3, 50),
    #     output_folder="Measured_cluster_results"
    # )
    #
    # feat = fv_both
    # cluster = pd.read_csv("Measured_cluster_results/Hierarchical_Cluster_Members.csv")
    # check_for_mismatch(feat, cluster)

#
# if __name__ == "__main__":
#     cas_df = pd.read_excel("House_data_compiled.xlsx")
#
#     # For SMILES
#     fv_aligned, cas_aligned = align_cas(feature_vectors, cas_df.copy())
#
#     # For Manual
#     mfeature_aligned, mcas_aligned = align_cas(mfeature_vectors, cas_df.copy())
#
#     # For Both
#     both_aligned, both_cas_aligned = align_cas(both_feature_vectors, cas_df.copy())
#
#     print("\n=== SMILES only clustering ===")
#     feature_vectors = pd.read_csv("SMILES_only_feature_vectors.csv")
#     fv_aligned, cas_aligned = align_cas(feature_vectors, cas_df.copy())
#
#     X = feature_vectors.iloc[:,1:].values
#     feature_names = feature_vectors.columns[1:].tolist()
#
#     main_clustering_pipeline(
#         X=X,
#         cas_df=cas_df,
#         feature_names=feature_names,
#         cluster_range=range(2, 50),  # number of clusters to test
#         output_folder="SMILES_only_cluster_results"  # folder to save all plots, CSVs, summaries
#     )
#
#     feat = pd.read_csv("SMILES_only_feature_vectors.csv")
#     cluster = pd.read_csv("SMILES_only_cluster_results/Hierarchical_Cluster_Members.csv")
#     check_for_mismatch(feat, cluster)
#
#     print("\n=== Manual only clustering ===")
#     mfeature_vectors = pd.read_csv("Manual_only_feature_vectors.csv")
#     mfeature_vectors, cas_df = align_cas(mfeature_vectors, cas_df)
#
#     X = mfeature_vectors.iloc[:, 1:].values
#     feature_names = mfeature_vectors.columns[1:].tolist()
#
#     main_clustering_pipeline(
#         X=X,
#         cas_df=cas_df,
#         feature_names=feature_names,
#         cluster_range=range(2, 50),  # number of clusters to test
#         output_folder="Manual_only_cluster_results"  # folder to save all plots, CSVs, summaries
#     )
#
#     feat = pd.read_csv("Manual_only_feature_vectors.csv")
#     cluster = pd.read_csv("Manual_only_cluster_results/Hierarchical_Cluster_Members.csv")
#     check_for_mismatch(feat, cluster)
#
#     print("\n=== Both clustering ===")
#     mfeature_vectors = pd.read_csv("Both_feature_vectors.csv")
#     mfeature_vectors, cas_df = align_cas(mfeature_vectors, cas_df)
#
#     X = mfeature_vectors.iloc[:, 1:].values
#     feature_names = mfeature_vectors.columns[1:].tolist()
#
#     main_clustering_pipeline(
#         X=X,
#         cas_df=cas_df,
#         feature_names=feature_names,
#         cluster_range=range(2, 50),  # number of clusters to test
#         output_folder="Both_cluster_results"  # folder to save all plots, CSVs, summaries
#     )
#
#     feat = pd.read_csv("Both_feature_vectors.csv")
#     cluster = pd.read_csv("Both_cluster_results/Hierarchical_Cluster_Members.csv")
#     check_for_mismatch(feat, cluster)
#
#


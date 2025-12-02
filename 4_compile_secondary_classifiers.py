import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def extract_concawe_matches(concawe_file: str, house_file: str, output_file: str = 'percent_comp_data.xlsx'):
    """
    Reads CAS numbers from the house data file, searches for matching rows in the Concawe table,
    and exports all matching Concawe rows to a new CSV file.
    The order of CAS numbers in the house file is preserved.
    """

    try:
        # Check file existence
        if not os.path.exists(concawe_file):
            raise FileNotFoundError(f"File not found: {concawe_file}")
        if not os.path.exists(house_file):
            raise FileNotFoundError(f"File not found: {house_file}")

        # Read Excel files
        concawe_df = pd.read_excel(concawe_file)
        house_df = pd.read_excel(house_file)

        # Find CAS columns (case-insensitive)
        concawe_cas_col = next((col for col in concawe_df.columns if 'cas' in col.lower()), None)
        house_cas_col = next((col for col in house_df.columns if 'cas' in col.lower()), None)

        if concawe_cas_col is None or house_cas_col is None:
            raise KeyError("Could not find a 'CAS' column in one or both Excel files.")

        # Clean up CAS numbers
        concawe_df[concawe_cas_col] = concawe_df[concawe_cas_col].astype(str).str.strip()
        house_df[house_cas_col] = house_df[house_cas_col].astype(str).str.strip()

        # Get list of CAS numbers from the house file
        cas_list = house_df[house_cas_col].dropna().unique().tolist()

        # Create an empty list to store matched rows
        matched_rows = []

        # Search each CAS in the Concawe data
        for cas in cas_list:
            matches = concawe_df[concawe_df[concawe_cas_col] == cas]
            if not matches.empty:
                matched_rows.append(matches)
            else:
                # Optionally: record CAS not found
                print(f"CAS {cas} not found in Concawe table.")



        # Combine all matched rows into one DataFrame
        if matched_rows:
            result_df = pd.concat(matched_rows, ignore_index=True)
        else:
            result_df = pd.DataFrame()
            print("No matches found between the two files.")

        sample_cols = [col for col in result_df.columns if 'sample number' in col.lower()]
        if sample_cols:
            result_df = result_df.drop(columns=sample_cols)
            print(f"Removed column(s): {', '.join(sample_cols)}")

        # Save to CSV
        result_df.to_excel(output_file, index=False)
        print(f"Successfully created '{output_file}' with {len(result_df)} records.")
        print(result_df)
    except Exception as e:
        print(f"An error occurred: {e}")

def count_manuals_per_cas(df):
    cas_manual_counts = []
    cas_list = df['CAS'].tolist()
    classifier_list = df['Manual classifiers'].tolist()
    delimiter = ','
    lengths = []
    for i, cas in enumerate(cas_list):
        classifier = str(classifier_list[i])
        if classifier != 'nan':
            print(str(classifier))
            if delimiter in classifier:
                classifiers = classifier.split(',')
                lengths.append(len(classifiers))
            else:
                lengths.append(1)

    print(lengths)
    print(len(lengths))

    # Count occurrences
    counts = Counter(lengths)

    # Sort by key (optional but makes the plot nicer)
    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    return x, y


from matplotlib.ticker import MaxNLocator


def plot_smiles_distribution(x, y, title="Distribution of manual classifiers per CAS"):
    plt.figure(figsize=(26, 20))

    # Bar plot
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
    plt.xlabel("Number of manually assigned classifiers", fontsize=65, labelpad=30)
    plt.ylabel("Number of UVCBs", fontsize=65, labelpad=30)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.grid(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_concawe_matches("concawe_table_fixed.xlsx", "house_data_compiled.xlsx")
    house = pd.read_excel("house_data_compiled.xlsx")
    x, y = count_manuals_per_cas(house)
    plot_smiles_distribution(x,y)

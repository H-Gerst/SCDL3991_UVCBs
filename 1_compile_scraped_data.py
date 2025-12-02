import pandas as pd
import os

# This code reads all of the individual excel files from ECHA CHEM for each UVCB in the House dataset.

def find_datafile(directory, cas_number):
    found_files = []
    for filename in os.listdir(directory):
        if cas_number in filename:
            found_files.append(filename)
            print(f'{filename} is available for {cas_number}')

    if len(found_files) == 1:
        ECHA_CHEM_file = found_files[0]

    else:
        print(f'There is no file with CAS number {cas_number}')
        return None

    return ECHA_CHEM_file


def extract_data_from_file(ECHA_CHEM_file, cas_to_search, substance_name, substance_category):
    identifiers = ECHA_CHEM_file.iloc[:,0].tolist()
    unique_identifiers = set(identifiers)
    empties = ['','*',' ','-', 'Not available', 'unknown', 'not available','not applicable']

    print(identifiers)
    print(unique_identifiers)
    structure_dict = {}

    for identifier in unique_identifiers:
        id_list = [identifier]
        i = 0
        while i < len(identifiers):
            if ECHA_CHEM_file.iloc[i,0] == identifier:
                item = ECHA_CHEM_file.iloc[i,1]
                if item in id_list or item in empties:
                    pass
                else:
                    id_list.append(ECHA_CHEM_file.iloc[i,1])
            i += 1

        # id_list_string = '; '.join(id_list[1:])
        id_list_string = '; '.join(str(item) for item in id_list[1:])

        structure_dict[id_list[0]] = id_list_string

    print(structure_dict)
    name_dict = {'CAS':f'{cas_to_search}', 'Substance name':f'{substance_name}', 'Category':f'{substance_category}'}
    name_dict.update(structure_dict)
    print(name_dict)
    return name_dict

def append_dict_to_excel(dictionary, excel_path):
    try:
        # Load the existing Excel file
        df = pd.read_excel(excel_path)

        # Ensure all expected columns are present in the new row
        new_row = {}
        for col in df.columns:
            if col in dictionary:
                new_row[col] = dictionary[col]
            else:
                new_row[col] = None  # or use "N/A" if you prefer

        # Create a DataFrame for the new row
        new_df = pd.DataFrame([new_row])

        # Append and save
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(excel_path, index=False)

        print(f"Dictionary data appended to {excel_path}")
        print(df)

    except Exception as e:
        print(f"Error updating Excel file: {e}")

# def append_dict_to_excel(dictionary, excel_path):
#     try:
#         # Load the existing Excel file
#         df = pd.read_excel(excel_path)
#         # headings = list(df.columns)
#         # df.loc[len(df)] = dictionary
#         new_df = pd.DataFrame([dictionary])
#         new_df = new_df[df.columns]
#         df = pd.concat([df, new_df], ignore_index=True)
#
#         print(df)
#         df.to_excel(excel_path, index=False)
#
#         print(f"Dictionary data appended to {excel_path}")
#
#     except Exception as e:
#         print(f"Error updating Excel file: {e}")

def read_cas_list(file, directory, collated_path):
    cas_list = file["CAS Number"].tolist()
    substance_name_list = file["Substance Name"].tolist()
    category_list = file["Category"].tolist()


    for i, cas_oi in enumerate(cas_list):
        substance_oi = substance_name_list[i]
        category_oi = category_list[i]
        print(f'Processing CAS: {cas_oi}|Substance: {substance_oi}')

        structure_file_name = find_datafile(directory, cas_oi)

        if structure_file_name:
            file_path = os.path.join(directory, structure_file_name)
            structure_file = pd.read_excel(file_path, header=None)  # Treat all rows as data
            UVCB_dict = extract_data_from_file(structure_file, cas_oi, substance_oi, category_oi)

            append_dict_to_excel(UVCB_dict, collated_path)
        else:
            print("No matching structure file found.")
            return

if __name__ == "__main__":
    UVCB_list = pd.read_excel('NIHMS1670764-supplement-Supplemental_Table_1.xlsx')
    UVCB_directory = 'C:/Users/hanna/PycharmProjects/SCDL3991'
    UVCBs_collated = 'C:/Users/hanna/PycharmProjects/SCDL3991/House_data_compiled.xlsx'

    read_cas_list(UVCB_list, UVCB_directory, UVCBs_collated)








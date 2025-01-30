import pandas as pd
import re

def format_name(name):
    match = re.match(r"([a-zA-Z]+)([A-Z][a-zA-Z]*)", name)
    if match:
        first_name = match.group(1)
        last_name = match.group(2)
        return f"{first_name} {last_name}"
    return name  
df = pd.read_csv('name_list.csv')

if 'name' not in df.columns:
    print("Error: 'name' column not found in the CSV file.")
else:
    df = df.dropna(subset=['name'])

    df['name'] = df['name'].str.lower()

    names_list = df['name'].tolist()
    names_list.sort(key=len, reverse=True)
    
    input_string = input("Enter name: ").lower()

    for name in names_list:
        if name in input_string:
            formatted_name = format_name(name)
            remaining_string = input_string.replace(name, "").strip()
            print(f"Name: {formatted_name} {remaining_string}")
            
            with open('formatted_names.txt', 'w') as file:
                file.write(f"Name: {formatted_name} {remaining_string}")
            break
    else:
        print(f"No names from the CSV file were found in the input string.")
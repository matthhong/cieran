# Transform all the .csv files in the current directory into a single .csv file

import csv
import glob
import os

# Get all the .csv files in the current directory
csv_files = glob.glob(os.path.join(os.getcwd(), '*.csv'))

# Header row
header = [
    "id",
    "color",
    "weight_0",
    "weight_1",
    "weight_2",
    "weight_3",
    "weight_4",
    "weight_5",
    "weight_6",
    "weight_7",
    "weight_8",
    "0_reward",
    "1_reward",
    "2_reward",
    "3_reward",
    "0",
    "1",
    "2",
    "3"
    ]

# Write the header row
with open('all.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Write the data rows
for csv_file in csv_files:
    with open(csv_file) as f:
        reader = csv.DictReader(f)

        # Check that this file has the same header as above
        if set(reader.fieldnames) != set(header):
            continue

        rows = []
        for row in reader:
            rows.append(row)

        # Remove duplicates (all values are the same)
        rows = [dict(t) for t in {tuple(d.items()) for d in rows}]
        rows.sort(key=lambda x: x['id'])

        # Write the rows to the new .csv file
        with open('all.csv', 'a') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row.values())
            
            
            # # Write the row to the new .csv file
            # with open('all.csv', 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(row.values())
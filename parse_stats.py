import sys
import os
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python parse_stats.py <file>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r') as file:
    lines = file.readlines()
    
# Initialize the dictionary to store test results
test_summaries = {}

df = pd.DataFrame(columns=['test_name', 'x_scope', 'y_scope', 'fence', 'order', 'weak_behaviors', 'rate'])

for line in lines:
    if 'Test' in line:
        parts = line.strip().split(' ')
        
        test_name = parts[1]
        
        test = test_name.split('-')
        
        # print(test)
        
        x_scope = test[2]
        y_scope = test[3]
        
        fence = test[4]
        order = test[6]
        
        weak_behaviors = int(parts[3].replace(',',''))
        rate = int(parts[-3])
        
        # Append the data to the DataFrame
        
        df.loc[len(df)] = {'test_name': test_name, 'weak_behaviors': weak_behaviors, 'rate': rate, 'x_scope': x_scope, 'y_scope': y_scope, 'fence': fence, 'order': order}

# Save the DataFrame to a CSV file
output_file = os.path.splitext(file_path)[0] + '_parsed.csv'
df.to_csv(output_file, index=False)

print(f"Parsed data saved to {output_file}")
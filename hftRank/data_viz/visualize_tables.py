import json
import os

import pandas as pd
from tabulate import tabulate
import pdfkit
import imgkit

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
# Define the path to wkhtmltoimage.exe
config = imgkit.config(wkhtmltoimage='C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe')
# read collated metrics data from results
df = pd.read_csv(f'../Results/{experiment_name}/collated_metrics_with_std.csv')

# Convert the DataFrame to a pretty table
table = tabulate(df, headers='keys', tablefmt='grid')

print(table)

latex_table = tabulate(df, headers='keys', tablefmt='latex')

print(latex_table)

# Save the LaTeX table to a file
with open(f'../Results/{experiment_name}/table.tex', 'w') as f:
    f.write(latex_table)
colors = ['#F00000','#3D6D9E', '#FFC725']

for color in colors:
    # Additional formatting with Pandas
    styled_table = df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', color), ('color', 'white')], }]
    ).set_properties(**{
        'border': '1px solid black',
        'padding': '5px'
    }).hide(axis='index')  # Corrected method

    # Export to HTML (which can be converted to other formats if needed)
    html_table = styled_table.to_html()

    # Save the styled table to an HTML file
    with open(f'../Results/{experiment_name}/styled_table_{color}.html', 'w') as f:
        f.write(html_table)

    # Convert HTML to PNG
    imgkit.from_file(f'../Results/{experiment_name}/styled_table_{color}.html', f'../Results/{experiment_name}/styled_table.png', config=config)

# Convert HTML to PDF pdfkit.from_file(f'../Results/{experiment_name}/styled_table.html', f'../Results/{
# experiment_name}/styled_table.pdf', configuration=config)

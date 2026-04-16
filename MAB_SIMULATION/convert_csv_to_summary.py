import pandas as pd
import glob
from pathlib import Path

csv_files = glob.glob("hit_ratio_analysis_*.csv")
if not csv_files:
    print("No CSV files found")
    exit()

latest_csv = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
df = pd.read_csv(latest_csv)

output_lines = []
for alpha in sorted(df['Alpha'].unique()):
    output_lines.append(f"\nALPHA = {alpha}")
    alpha_df = df[df['Alpha'] == alpha]
    for _, row in alpha_df.iterrows():
        alg = row['Algorithm']
        hit = row['overall_hit_ratio']
        output_lines.append(f"{alg} {hit:.4f}")

with open("comprehensive_summary_generated.txt", "w") as f:
    f.write("\n".join(output_lines))

print("✅ Converted to comprehensive_summary_generated.txt")
import pandas as pd
import sys

def explore():
    path = r'c:\Users\DELL\Downloads\student_resource\dataset\train\train_source1.tsv'
    # Just read the country column to avoid memory issues
    df = pd.read_csv(path, sep='\t', usecols=['country'])
    dist = df['country'].value_counts(dropna=False)
    print("Source 1 Country Distribution:")
    print(dist)

if __name__ == "__main__":
    explore()

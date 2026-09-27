import pandas as pd
import time
t0 = time.time()
df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
print(f"Read {len(df)} rows in {time.time()-t0:.2f}s")

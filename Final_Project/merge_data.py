import pandas as pd
import numpy as np
import os

NORMAL_FILE = 'normal.csv'
ICMP_FILE = 'icmp_attack.csv'
SYN_FILE = 'syn_attack.csv'
OUTPUT_FILE = 'final_multiclass_dataset.csv'


try:
    df_normal = pd.read_csv(NORMAL_FILE)
    df_icmp = pd.read_csv(ICMP_FILE)
    df_syn = pd.read_csv(SYN_FILE)
    print("Files loaded successfully: Normal, ICMP, and SYN.")
except FileNotFoundError as e:
    print(f" Error: Required file not found. Please ensure {e.filename} exists.")
    exit()


# 0 = Normal Traffic
# 1 = ICMP Flood Attack
# 2 = SYN Flood Attack
df_normal['Label'] = 0
df_icmp['Label'] = 1
df_syn['Label'] = 2

df_final = pd.concat([df_normal, df_icmp, df_syn], ignore_index=True)


df_final = df_final.fillna(0)


df_final.to_csv(OUTPUT_FILE, index=False)

print("-" * 50)
print(f" Success! Multiclass dataset created: '{OUTPUT_FILE}'")
print(f"Total Records: {len(df_final)}")
print(f"Normal (0) Records: {len(df_normal)}")
print(f"ICMP (1) Records: {len(df_icmp)}")
print(f"SYN (2) Records: {len(df_syn)}")
print("-" * 50)

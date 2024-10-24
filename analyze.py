import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE_PREFIX = "vllm_benchmarking_Llama31_8B_"
NUM_FILES = 1

df = pd.DataFrame()
BATCH_SIZES = [1, 4, 8]
PROMPT_LENGTHS = [128, 256, 512, 2048, 4096]

iter_num = 0
for bs in BATCH_SIZES:
    for pl in PROMPT_LENGTHS:
        file_name = (
            DATA_FILE_PREFIX + "bs_" + str(bs) + "_pl_" + str(pl) + ".csv"
        )
        if iter_num == 0:
            df = pd.read_csv(file_name)
        else:
            data = pd.read_csv(file_name)
            df = pd.concat([df, data], ignore_index=True)
        iter_num += 1

        # Sort by batch size, followd by input length and generation length

df = df.sort_values(by=["batch_size", "prompt_length", "generation_length"])

# Plot the tokens_per_second for each batch size as a line plot with different prompt lengths
# on the same plot

df = df[df["generation_length"] == 1024]
sns.lineplot(df, x="batch_size", y="tokens_per_sec", hue="prompt_length")

plt.savefig("tokens_per_second.png")

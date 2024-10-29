import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from setup_plot import *
import numpy as np

#matplotlib.use("TkAgg")

setup()
setup_global()

FOLDER_PATH = "Llama31_8B_TP2/"

DATA_FILE_PREFIX = "vllm_benchmarking_Llama31_8B_"
NUM_FILES = 1

df = pd.DataFrame()
BATCH_SIZES = [1, 4, 8, 16, 32]
PROMPT_LENGTHS = [128, 256, 512, 2048, 4096]

iter_num = 0
for bs in BATCH_SIZES:
    for pl in PROMPT_LENGTHS:
        file_name = (
            FOLDER_PATH + DATA_FILE_PREFIX + "bs_" + str(bs) + "_pl_" + str(pl) + ".csv"
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
ax = plot_line_chart(df, x="batch_size", 
                y="tokens_per_ms", 
                hue="prompt_length",
                xlabel="Batch Size", 
                ylabel="Tokens per MilliSecond", 
                title=f"Tokens per MilliSecond vs Batch Size",
                marker="o",
                markersize=14,
)
ax.set_ylim(0, 1.25)
plt.savefig(FOLDER_PATH + "Throuhgput_BatchSize.png")

ax = plot_line_chart(df, x="batch_size", 
                y="tokens_per_ms", 
                hue="generation_length",
                xlabel="Batch Size", 
                ylabel="Tokens per MilliSecond", 
                title=f"Tokens per MilliSecond vs Batch Size",
                marker="o",
                markersize=14,
)
ax.set_ylim(0, 1.25)
plt.savefig(FOLDER_PATH + "Throuhgput_BatchSize_GenerationLength.png")


# For each batch size, plot the tokens_per_ms for each prompt length as a line plot with x axis as the generation length
# and y axis as tokens_per_ms
for bs in BATCH_SIZES:
    df_bs = df[df["batch_size"] == bs]
    ax = plot_line_chart(df_bs, x="generation_length", 
                    y="tokens_per_ms", 
                    hue="prompt_length",
                    xlabel="Generation Length", 
                    ylabel="Tokens per MilliSecond", 
                    title=f"Tokens per MilliSecond vs Generation Length\nBatch Size = {bs}",
                    marker="o",
                    markersize=14,
    )
    ax.set_ylim(0, 1.25)
    plt.savefig(FOLDER_PATH + f"Throuhgput_GenerationLength_bs_{bs}.png")

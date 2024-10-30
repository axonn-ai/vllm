import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from setup_plot import *
import numpy as np

#matplotlib.use("TkAgg")

setup()
setup_global()

FOLDER_PREFIX = "Llama31_8B_TP"

TP_SIZES = [1, 2, 4, 8]
DATA_FILE_PREFIX = "vllm_benchmarking_Llama31_8B_"
NUM_FILES = 1

BATCH_SIZES = [1, 4, 8, 16, 32]
PROMPT_LENGTHS = [128, 256, 512, 2048, 4096]

df = pd.DataFrame()
iter_num = 0
for tp in TP_SIZES:
    FOLDER_PATH = FOLDER_PREFIX + str(tp) + "/"
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
    
    df = df.sort_values(by=["tensor_parallel_size", "batch_size", "prompt_length", "generation_length"])


# We want to analyze the throughput of the model with batch size on the x-axis, TP size as the hue, prompt length as the style, and tokens per second as the y-axis

fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.tight_layout(pad=4) 

gen_len = 2048

for i, len in enumerate(PROMPT_LENGTHS[1:]):
    r = i // 2
    c = i % 2
    ax = axs[r, c]

    df_plot = df[df["prompt_length"] == len]
    df_plot = df_plot[df_plot["generation_length"] == gen_len]

    # Change the tensor_parallel_size to be string so that it is displayed as a category
    df_plot["tensor_parallel_size"] = df_plot["tensor_parallel_size"].astype(str)

    print (df_plot)
    ax = plot_line_chart(df_plot, x="tensor_parallel_size",
                    y="tokens_per_ms",
                    hue="batch_size",
                    xlabel="#GCDs",
                    ylabel="Tokens per MilliSecond",
                    title=f"Tokens per MilliSecond vs TP Degree (PromptLength = {len})",
                    marker="o",
                    markersize=14,
                    ax=ax,
    )
    # Set y axis to log scale
    #ax.set_yscale("log")
    
    #ax.set_xticks(TP_SIZES)
    #ax.set_ylim(0, 1.25)
fig.suptitle(f"Tokens per MilliSecond vs TP Degree (Generation Length= {gen_len})", fontsize=16)
  
fig.savefig(f"bench_plots/Scaling_genl_{gen_len}.pdf", bbox_inches='tight')

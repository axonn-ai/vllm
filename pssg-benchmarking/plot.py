import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

files = sys.argv[1].split(",")
df = pd.concat(
    pd.read_csv(f, delimiter = ",")
    for f in files
)
df["tokens_per_sec"] = df["tokens_per_ms"].map(lambda x: x * 1000)
gdf = df.groupby("model")
models = list(gdf.indices.keys())
for model in models:
    mdf = gdf.get_group(model)
    gdf_prompt_lens = mdf.groupby("prompt_length")
    prompt_lens = list(gdf_prompt_lens.indices.keys())
    out_f_root = model.replace("/", "_")
    out_f_root = out_f_root.replace(".", "_")
    for pl in prompt_lens:
        data = gdf_prompt_lens.get_group(pl)
        title = model + " (Prompt Length: " + str(pl) + ")"
        sns.set(style = "whitegrid")
        plt.figure(figsize = (10,6))
        print(data["prompt_length"])
        sns.lineplot(
            data = data,
            x = "generation_length",
            y = "tokens_per_sec",
            hue = "batch_size",
            marker = "o",
            markers = True,
            dashes = False
        )
        plt.legend(title = "Batch Size", loc = "upper right")
        plt.title(title)
        plt.ylabel("Tokens Per Second")
        plt.xlabel("Generation Length")
        plt.savefig(out_f_root + "_prompt" + str(pl) + ".png")


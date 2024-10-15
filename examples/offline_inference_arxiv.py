from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("ccdv/arxiv-summarization", "section", split="test")
prompts = ds["article"]
batch_size = 1

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

# Create an LLM.
llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=4, enforce_eager=True)

# Iterate over the prompts in batches of size `batch_size`.
for i in tqdm(range(0, len(prompts), batch_size)):
    # Get the prompts for the current batch.
    batch_prompts = prompts[i : i + batch_size]

    outputs = llm.generate(batch_prompts, sampling_params)

    # Print the outputs.
    #for output in outputs:
    #    prompt = output.prompt
    #    generated_text = output.outputs[0].text
    #    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

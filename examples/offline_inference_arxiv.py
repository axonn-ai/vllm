from vllm import LLM, SamplingParams, TokensPrompt
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import configparser
from datetime import datetime

config = configparser.ConfigParser()
config.read(sys.argv[1])

def from_csv(s) -> list[int]:
    return list(map(int, s.split(","))) if s else None

dataset = config["DEFAULT"]["dataset"]
batch_sizes = from_csv(config["DEFAULT"]["batch_sizes"])
prompt_lengths = from_csv(config["DEFAULT"]["prompt_lengths"])
generation_lengths = from_csv(config["DEFAULT"]["generation_lengths"])
tensor_parallel_sizes = from_csv(config["DEFAULT"]["tensor_parallel_sizes"])
num_warmups = int(config["DEFAULT"]["num_warmups"])
num_iterations = int(config["DEFAULT"]["num_iterations"])
output_file = config["DEFAULT"]["output_file"]
model = config["DEFAULT"]["model"]
eager_mode = config.getboolean("DEFAULT", "eager_mode", fallback=False)

print (config)


ds = load_dataset(dataset, "section", split = "test")
prompts = ds[config["DEFAULT"]["dataset_key"]]
#prompts = prompts[:1000]

def write_out(line: str, mode: str = "a"):
    with open(output_file, mode) as file:
        file.write(line + "\n")

write_out(
    "model,dataset,tensor_parallel_size,batch_size,"
    "prompt_length,generation_length,tokens_per_sec",
    "w"
)

def generate_with_params(llm, sampling_params, prompts, prompt_length, batch_size, generation_length) -> list[int]:
    tokenizer = llm.get_tokenizer()

    time_generating_tokens = 0.0
    tokens_generated = 0

    prompts = prompts[:(num_warmups + num_iterations) * batch_size]
    iter_num = 0
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        batch_prompts_tokenized = []
        for text in batch_prompts:
            ids = tokenizer.encode(text)[:prompt_length] # warns bc result of encode is too long, not an issue bc truncated after
            if len(ids) < prompt_length:
               ids += ([tokenizer.eos_token_id] * (prompt_length - len(ids)))
            tp = TokensPrompt(prompt_token_ids = ids)
            batch_prompts_tokenized.append(tp)
        outputs = llm.generate(batch_prompts_tokenized, sampling_params)

        duration = 0
        output_tokens_per_batch = 0
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            token_count = len(token_ids)
            metrics = output.metrics
            start = metrics.first_token_time
            end = metrics.finished_time
            duration += (end - start)
            output_tokens_per_batch += token_count

        duration = float(duration) / len(outputs)
        if iter_num < num_warmups: # Skip the warmup iterations
            pass
        else:   
            # TODO: Verify this. Essentially if we are processing a batch each duration should be exactly the same for the outputs of a single batch since the batch is processed in parallel
            time_generating_tokens += duration
            tokens_generated += output_tokens_per_batch

        iter_num += 1

    # TODO: I am not sure if we should find throughput per batch adn then average it across all iterations or do this
    tokens_per_sec = float(tokens_generated) / time_generating_tokens
    return tokens_per_sec

for ts in tensor_parallel_sizes:
    llm = LLM(
        model = model,
        tensor_parallel_size = ts,
        enforce_eager = eager_mode
    )
    for gl in generation_lengths:
        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = gl,
            min_tokens = gl
        )
        for pl in prompt_lengths:
            for bs in batch_sizes:
                result = generate_with_params(
                    llm,
                    sampling_params,
                    prompts,
                    pl,
                    bs,
                    gl
                )
                line = model + "," + dataset + "," + \
                        str(ts) + "," + \
                        str(bs) + "," + str(pl) + "," + \
                        str(gl) + "," + str(result)
                write_out(line)

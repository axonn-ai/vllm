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
output_file = config["DEFAULT"]["output_file"]
model = config["DEFAULT"]["model"]

ds = load_dataset(dataset, "section", split = "test")
prompts = ds[config["DEFAULT"]["dataset_key"]]
prompts = prompts[:5]

def write_out(line: str):
    with open(output_file, "w") as file:
        file.write(line + "\n")

write_out(
    "model,dataset,tensor_parallel_size,batch_size,"
    "prompt_length,generation_length,avg_ttft_from_arrive,"
    "avg_ttft_from_sched,avg_throughput_sched_to_last,"
    "avg_throughput_first_to_last,avg_throughput_arrive_to_last,avg_total"
)

def generate_with_params(llm, sampling_params, prompts, prompt_length, batch_size, generation_length) -> list[int]:
    tokenizer = llm.get_tokenizer()

    avg_ttft_from_arrive = 0.0
    avg_ttft_from_sched = 0.0
    avg_throughput_sched_to_last = 0.0
    avg_throughput_first_to_last = 0.0
    avg_throughput_arrive_to_last = 0.0
    avg_total = 0.0
    total_prompts = 0

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        print("made it here")
        print(generation_length)
        batch_prompts_tokenized = [
            TokensPrompt(prompt_token_ids = tokenizer.encode(text)[:prompt_length])
            for text in batch_prompts
        ]
        #batch_prompts = [prompt[:200] for prompt in batch_prompts]
        outputs = llm.generate(batch_prompts_tokenized, sampling_params)
        total_prompts += len(outputs)
        
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            token_count = len(token_ids)
            if token_count != generation_length: print("WARNING: token count != generation length")

            metrics = output.metrics
            print(metrics)
            arrival = metrics.arrival_time
            scheduled = metrics.first_scheduled_time
            first_token = metrics.first_token_time
            last_token =  metrics.last_token_time
            finished = metrics.finished_time

            avg_ttft_from_arrive += first_token - arrival
            avg_ttft_from_sched += first_token - scheduled
            avg_throughput_sched_to_last += (token_count / (last_token - scheduled))
            print(last_token - first_token)
            avg_throughput_first_to_last += (token_count / (last_token - first_token))
            avg_throughput_arrive_to_last += (token_count / (last_token - arrival))
            avg_total += finished - arrival

    avg_ttft_from_arrive /= total_prompts
    avg_ttft_from_sched /= total_prompts
    avg_throughput_sched_to_last /= total_prompts
    avg_throughput_first_to_last /= total_prompts
    avg_throughput_arrive_to_last /= total_prompts
    avg_total /= total_prompts

    return [
        avg_ttft_from_arrive, avg_ttft_from_sched,
        avg_throughput_sched_to_last, avg_throughput_first_to_last,
        avg_throughput_arrive_to_last, avg_total
    ]

for ts in tensor_parallel_sizes:
    llm = LLM(
        model = model,
        tensor_parallel_size = ts,
        enforce_eager = True
    )
    for gl in generation_lengths:
        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = gl,
            min_tokens = gl
        )
        for pl in prompt_lengths:
            # TODO: pad prompts to exact prompt length, for now filter out those that aren't
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
                        str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + \
                        str(result[4]) + "," + str(result[5]) + ","
                write_out(line)

from vllm import LLM, SamplingParams, TokensPrompt
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import configparser
from datetime import datetime
import argparse


def from_csv(s) -> list[int]:
    return list(map(int, s.split(","))) if s else None


def write_out(output_file, line: str, mode: str = "a"):
    with open(output_file, mode) as file:
        file.write(line + "\n")


def generate_with_params(
    llm,
    sampling_params,
    prompts,
    prompt_length,
    batch_size,
    generation_length,
    num_warmups,
    num_iterations,
) -> list[int]:
    tokenizer = llm.get_tokenizer()

    time_generating_tokens = 0.0
    tokens_generated = 0
    throughput_per_batch = 0

    prompts = prompts[: (num_warmups + num_iterations) * batch_size]
    iter_num = 0
    num_valid_iters = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        batch_prompts_tokenized = []
        for text in batch_prompts:
            ids = tokenizer.encode(
                text
            )[
                :prompt_length
            ]  # warns bc result of encode is too long, not an issue bc truncated after
            if len(ids) < prompt_length:
                ids += [tokenizer.eos_token_id] * (prompt_length - len(ids))
            tp = TokensPrompt(prompt_token_ids=ids)
            batch_prompts_tokenized.append(tp)

        # TODO: Verify that vLLM processes prompts in parallel and does not break the batch
        start.record()
        outputs = llm.generate(batch_prompts_tokenized, sampling_params)
        end.record()

        torch.cuda.synchronize()

        # TODO: Need more fine-grained metrics going forward. vLLM does seem to be breaking prompts into smaller chunks. Until we verify that, this is a good estimate for duration to calculate throughput
        batch_duration = start.elapsed_time(end)

        duration = 0
        output_tokens_per_batch = 0
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            token_count = len(token_ids)
            metrics = output.metrics
            start = metrics.first_token_time
            end = metrics.finished_time
            duration += end - start
            output_tokens_per_batch += token_count

        ### Average the duration over all prompts in the batch (all prompts are processed in parallel ideally)
        #batch_duration = float(duration) / len(outputs)

        if iter_num < num_warmups:  # Skip the warmup iterations
            pass
        else:
            throughput_per_batch += (
                float(output_tokens_per_batch) / batch_duration
            )

            num_valid_iters += 1
        iter_num += 1

    assert num_valid_iters == num_iterations
    avg_throughput_per_batch = float(throughput_per_batch) / num_iterations
    return avg_throughput_per_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking for vLLM")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file",
        required=True,
    )
    parser.add_argument(
        "-pl",
        "--prompt_lengths",
        nargs="+",
        type=int,
        help="Prompt lengths (overrides the config)",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output CSV File Path",
        required=False,
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    default_config = config["DEFAULT"]

    if args.prompt_lengths is not None:
        default_config["prompt_lengths"] = ",".join(
            map(str, args.prompt_lengths)
        )
    if args.output_file is not None:
        default_config["output_file"] = args.output_file

    print(default_config)

    ds = load_dataset(default_config["dataset"], "section", split="test")
    prompts = ds[default_config["dataset_key"]]

    outfile = default_config["output_file"]
    write_out(
        outfile,
        "model,dataset,tensor_parallel_size,batch_size,"
        "prompt_length,generation_length,tokens_per_sec",
        "w",
    )

    for ts in from_csv(default_config["tensor_parallel_sizes"]):
        llm = LLM(
            model=default_config["model"],
            tensor_parallel_size=ts,
            enforce_eager=default_config.getboolean(
                "eager_mode", fallback=False
            ),
            #enable_chunked_prefill=False,
        )
        for bs in from_csv(default_config["batch_sizes"]):
            for pl in from_csv(default_config["prompt_lengths"]):
                for gl in from_csv(default_config["generation_lengths"]):
                    sampling_params = SamplingParams(
                        temperature=0.8,
                        top_p=0.95,
                        max_tokens=gl,
                        min_tokens=gl,
                    )
                    result = generate_with_params(
                        llm,
                        sampling_params,
                        prompts,
                        pl,
                        bs,
                        gl,
                        num_warmups=default_config.getint("num_warmups"),
                        num_iterations=default_config.getint("num_iterations"),
                    )
                    line = (
                        default_config["model"]
                        + ","
                        + default_config["dataset"]
                        + ","
                        + str(ts)
                        + ","
                        + str(bs)
                        + ","
                        + str(pl)
                        + ","
                        + str(gl)
                        + ","
                        + str(result)
                    )
                    write_out(outfile, line)

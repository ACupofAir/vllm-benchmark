"""
usage: python ./client.py -c 4 -o 512 -s
	-c N: N reqs
	-o n: n outputs
"""
import openai
import time
import argparse
import concurrent.futures
import requests


llamacpp_base_url="http://localhost:8001/v1"
llamacpp_api_key="sk-no-key-required"
llamacpp_model="davinci-002"


def run_single_request(index, max_tokens=1024, stream=False):
    client = openai.OpenAI(
        base_url=llamacpp_base_url,
        api_key=llamacpp_api_key
    )

    response = client.completions.create(
        model=llamacpp_model,
        prompt=(
            "A conversation between User and Assistant. The user asks a question,"
            "and the Assistant solves it. The assistant first thinks about the reasoning process"
            "in the mind and then provides the user with the answer. The reasoning process and answer"
            "are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning"
            "process here </think> <answer> answer here </answer>. User: Question: If ( a > 1 ), then the sum of the"
            "real solutions of ( sqrt{a} - sqrt{a + x} = x ) is equal to:. Assistant: <think>"
        ),
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:

        print(f"---------- start of stream {index} --------- ")
        output = ""
        for chunk in response:
            text = chunk.choices[0].text
            print(f"[stream ({index}):{text}]", end="", flush=True)
            output += text
        print(f"---------- end of stream {index} --------- ")  # newline after stream
    else:
        print("-------------- non-stream output Start -------------")
        print(response.choices[0].text)
        print("-------------- non-stream output End --------------")
        print("--- Timings ---")
        print(response.timings)

def client_concurrent(concurrency=4, max_tokens=1024, stream=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_request, i, max_tokens=max_tokens, stream=stream)
            for i in range(concurrency)
        ]
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llama.cpp serve")
    parser.add_argument("-c","--concurrency", type=int, default=0, help="Number of concurrent requests")
    parser.add_argument("-s","--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("-o", "--out-tokens", type=int, default=32, help="Number of output tokens")
    args = parser.parse_args()
    if (args.concurrency > 0):
        client_concurrent(args.concurrency, args.out_tokens, args.stream)

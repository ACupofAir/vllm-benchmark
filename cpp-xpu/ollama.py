import fire
from io import StringIO
from datetime import datetime, UTC
import requests
import json
import sys
import signal

benchmark_terminating = False

def benchmark_context(*, port=11434, window=24*1024, utilization=0.8, model='qwen2.5-coder:7b'):
    def signal_handler(sig, frame):
        global benchmark_terminating
        if benchmark_terminating:
            print("Benchmark was terminated mid-iteration.")
            sys.exit(0)
        print("Finishing current iteration...")
        benchmark_terminating = True
    signal.signal(signal.SIGINT, signal_handler)
    counter = 1
    while not benchmark_terminating:
        length = 0
        buffer = StringIO()
        while length < utilization * window:
            line = str(counter) + '\n'
            buffer.write(line)
            length += len(line)
            counter += 1
        messages = [{
            'role': 'user',
            'content': buffer.getvalue() + 'Next?'
        }]
        buffer.close()
        for i in range(5):
            messages.append({
                'role': 'assistant',
                'content': str(counter)
            })
            counter += 1
            messages.append({
                'role': 'user',
                'content': 'Next?'
            })
        http_response = requests.post(f'http://localhost:{port}/api/chat', json={
            'model': model,
            'options': {
                'num_ctx': window
            },
            'stream': False,
            'messages': messages
        })
        http_response.raise_for_status()
        response = http_response.json()
        response_text = response['message']['content']
        correct = str(counter) == response_text
        if not correct:
            print(f'Incorrect: {response_text} (expected {counter})')
        counter += 1
        prompt_tokens = response['prompt_eval_count']
        prompt_nanos = response['prompt_eval_duration']
        print(f'{prompt_tokens/1024:,.1f}K @ {prompt_tokens / (prompt_nanos * 1e-9):,.0f} t/s')

if __name__ == '__main__':
    fire.Fire()

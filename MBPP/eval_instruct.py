import argparse
import json
import os
import torch
import re
from pathlib import Path
from tqdm import tqdm
from groq import Groq
import requests
import time
import os

data_abs_dir = Path(__file__).parent / "data"

from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }

def convert_for_evaluation(example):
    gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example['generation'] = generation
    return example

def generate_one(example, model):
    prompt = example['prompt']
    messages = [{'role': 'user', 'content': prompt }]
    if model == "Groq":
        client = Groq(
            api_key=os.environ['GROQ_API_KEY'],
        )
        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="llama3-8b-8192",
                    temperature=0
                )
                example['gpt_completion'] = chat_completion.choices[0].message.content
            except:
                time.sleep(5)
                continue
            break
    elif model == "Samba":
        payload = {
            "inputs": messages,
            "params": {
                "max_tokens_allowed_in_completion": {"type": "int", "value": 500},
                "min_token_capacity_for_completion": {"type": "int", "value": 2},
                "skip_special_token": {"type": "bool", "value": True},
                "stop_sequences": {"type": "list", "value": ["[INST]", "[INST]", "[/INST]", "[/INST]"]}
            },
            "expert": "llama3-8b"
        }
        url = os.environ.get("SAMBA_URL")
        key = os.environ.get("SAMBA_KEY")

        headers = {
            "Authorization": f"Basic {key}",
            "Content-Type": "application/json"
        }
        post_response = requests.post(url, json=payload, headers=headers, stream=True)

        response_text = ""
        for line in post_response.iter_lines():
            if line.startswith(b"data: "):
                data_str = line.decode('utf-8')[6:]
                try:
                    line_json = json.loads(data_str)
                    content = line_json.get("stream_token", "")
                    if content:
                        response_text += content
                except json.JSONDecodeError as e:
                    pass
        example['gpt_completion'] = response_text
    return convert_for_evaluation(example)

def generate_main(args):
    model = args.model
    saved_path = args.output_path
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    print("model", model)

    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []

    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, model)
        generated_examples.append(gen_example)
        print("Generate {}/{} over...".format(len(generated_examples), len(examples)))

    print("Generate all over!!!")
    
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp_test.jsonl"),
        language='python',
        is_mbpp=True
    )
    print(result, model)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
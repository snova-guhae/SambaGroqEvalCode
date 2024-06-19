import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
import openai
import time
import numpy as np
from groq import Groq
import requests

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, languge_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness

def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def generate_one(example, lang, model):
    prompt = build_deepseekcoder_instruction(languge_settings[lang]['full_name'], example['prompt'])

    messages = [{'role': 'user', 'content': prompt }]
   
    if model == "groq":
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0
        )
        
        example['output'] = chat_completion.choices[0].message.content
    elif model == "samba":
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
        post_response = requests.post(f'https://{url}/api/v1/chat/completion', json=payload, headers=headers, stream=True)

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

        example['output'] = response_text
    return extract_generation_code(example, lang_code=lang)

def generate_main(args):
    model = args.model
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    print("model", model)
    
    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    times = []
    for ex in tqdm(examples, desc='Generating'):
        t0 = time.time()
        gen_example = generate_one(ex, args.language, model)
        t1 = time.time()
        times.append(t1 - t0)
        generated_examples.append(gen_example)
    print(f"TOTAL TIME: {np.mean(times)}")

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result, model)
    pass

def evaluation_only(args):
    lang = args.language
    temp_dir = args.temp_dir
    assert os.path.exists(args.output_path), "Not fond output file: {}".format(args.output_path)
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.basename(args.output_path)
    output_examples = [json.loads(x) for x in open(args.output_path) if x.strip()]

    processed_examples = [extract_generation_code(ex, lang) for ex in tqdm(output_examples, "Processing")]
    processed_path = os.path.join(temp_dir, output_name)
    with open(processed_path, 'w', encoding='utf-8') as fw:
        for ex in processed_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(processed_examples), processed_path))

    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    from human_eval.evaluation import evaluate_functional_correctness
    result = evaluate_functional_correctness(
        input_file=processed_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--language', type=str, help="langauge")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass

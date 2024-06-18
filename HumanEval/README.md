# Instructions to recreate HumanEval results
Example code is in eval_instruct.sh.
Set your keys and URL. 
```bash
    export SAMBA_URL=<INSERT SAMBA URL>
    export SAMBA_KEY=<INSERT SAMBA KEY>
    export GROQ_API_KEY=<INSERT GROQ KEY>
```

Then choose your language and run the eval. Use "samba" model name for Samba-1 and "groq" model name for Groq
```bash 
    export SAMBA_URL=<INSERT SAMBA URL>
    export SAMBA_KEY=<INSERT SAMBA KEY>
    export GROQ_API_KEY=<INSERT GROQ KEY>

    python eval_instruct.py --model groq --language python --output_path ./groq.json --temp_dir ./
    python eval_instruct.py --model samba --language python --output_path ./samba.json --temp_dir ./
```

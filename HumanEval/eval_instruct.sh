export SAMBA_URL=<INSERT SAMBA URL>
export SAMBA_KEY=<INSERT SAMBA KEY>
export GROQ_API_KEY=<INSERT GROQ KEY>

python eval_instruct.py --model groq --language python --output_path ./groq.json --temp_dir ./
python eval_instruct.py --model samba --language python --output_path ./samba.json --temp_dir ./
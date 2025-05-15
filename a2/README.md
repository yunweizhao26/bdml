
```
echo 'export TRANSFORMERS_CACHE=/workspace/work/cache/' >> ~/.bashrc
echo 'export HF_HOME=/workspace/work/cache/' >> ~/.bashrc
echo 'export XDG_CACHE_HOME=/workspace/work/cache/' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/workspace/work/cache/' >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/workspace/work/a2

echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init

pip install -r requirements.txt
pip install flash-attn --no-build-isolation

python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py     --input_dir /scratch/yz5944/bdml/a1/Llama3.2-3B/ --model_size 3B --llama_version 3.2 --output_dir /scratch/yz5944/bdml/a1/models/Llama3.2-3B
'''

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
/workspace/miniconda3
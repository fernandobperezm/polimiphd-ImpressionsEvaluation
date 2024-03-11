poetry lock --no-update
poetry install --sync
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install --no-use-pep517 lightfm

# Installs the wheel compatible with CUDA 11.8. See: https://pytorch.org/get-started/locally/
poetry run pip install --upgrade torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Installs the wheel compatible with CUDA 11.8 and cuDNN 8.6 or newer. See: https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder
poetry run pip install --upgrade "jax[cuda11_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


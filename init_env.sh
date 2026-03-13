conda create -n torchjax python=3.11 -y
conda activate torchjax
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
pip install "jax==0.4.28"
pip install --no-deps \
  https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.28+cuda12.cudnn89-cp311-cp311-manylinux2014_x86_64.whl
pip install --no-deps "chex==0.1.91" "optax==0.2.4" "flax==0.10.2" "evosax==0.1.6"
pip install pyyaml rich matplotlib dotmap msgpack msgpack-numpy
pip install --no-deps "orbax-checkpoint==0.5.16" "tensorstore==0.1.45"
pip install \
  absl-py==2.3.1 aiofiles==25.1.0 annotated-types==0.7.0 anyio==4.11.0 appnope==0.1.4 \
  argon2-cffi==25.1.0 argon2-cffi-bindings==25.1.0 arrow==1.4.0 asttokens==3.0.0 async-lru==2.0.5 \
  attrs==25.4.0 babel==2.17.0 beautifulsoup4==4.14.2 bleach==6.3.0 bokeh==3.8.0 certifi==2025.10.5 \
  cffi==2.0.0 charset-normalizer==3.4.4 click==8.3.0 comm==0.2.3 contourpy==1.3.3 cycler==0.12.1 \
  debugpy==1.8.17 decorator==4.4.2 defusedxml==0.7.1 dotmap==1.3.30 einops==0.8.1 \
  etils==1.13.0 exceptiongroup==1.3.0 executing==2.2.1 fastjsonschema==2.21.2 filelock==3.20.0 \
  fonttools==4.60.1 fqdn==1.5.1 fsspec==2025.10.0 gitdb==4.0.12 GitPython==3.1.45 h11==0.16.0 \
  hf-xet==1.2.0 httpcore==1.0.9 httpx==0.28.1 huggingface-hub==0.36.0 humanize==4.14.0 idna==3.11 \
  ImageIO==2.37.2 imageio-ffmpeg==0.6.0 importlib_resources==6.5.2 ipykernel==7.1.0 ipython==9.6.0 \
  ipython_pygments_lexers==1.1.1 ipywidgets==8.1.8 isoduration==20.11.0 jedi==0.19.2 Jinja2==3.1.6 \
  json5==0.12.1 jsonpointer==3.0.0 jsonschema==4.25.1 jsonschema-specifications==2025.9.1 \
  jupyter-events==0.12.0 jupyter-lsp==2.3.0 jupyter_client==8.6.3 jupyter_core==5.9.1 \
  jupyter_server==2.17.0 jupyter_server_terminals==0.5.3 jupyterlab==4.4.10 jupyterlab_pygments==0.3.0 \
  jupyterlab_server==2.28.0 jupyterlab_widgets==3.0.16 kiwisolver==1.4.9 lark==1.3.1 \
  markdown-it-py==4.0.0 MarkupSafe==3.0.3 matplotlib==3.10.7 matplotlib-inline==0.2.1 mdurl==0.1.2 \
  mistune==3.1.4 ml_dtypes==0.5.3 moviepy==1.0.3 mpmath==1.3.0 msgpack==1.1.2 narwhals==2.10.1 \
  nbclient==0.10.2 nbconvert==7.16.6 nbformat==5.10.4 nest-asyncio==1.6.0 networkx==3.5 \
  notebook_shim==0.2.4 opt_einsum==3.4.0 overrides==7.7.0 packaging==25.0 pandas==2.3.3 \
  pandocfilters==1.5.1 parso==0.8.5 pexpect==4.9.0 pillow==12.0.0 platformdirs==4.5.0 plotly==6.3.1 \
  proglog==0.1.12 prometheus_client==0.23.1 prompt_toolkit==3.0.52 protobuf==6.33.0 psutil==7.1.2 \
  ptyprocess==0.7.0 pure_eval==0.2.3 pycparser==2.23 pydantic==2.12.3 pydantic_core==2.41.4 \
  Pygments==2.19.2 pyparsing==3.2.5 python-dateutil==2.9.0.post0 python-dotenv==1.2.1 \
  python-json-logger==4.0.0 pytz==2025.2 pyzmq==27.1.0 rdkit==2025.9.1 referencing==0.37.0 \
  regex==2025.10.23 requests==2.32.5 rfc3339-validator==0.1.4 rfc3986-validator==0.1.1 \
  rfc3987-syntax==1.1.0 rich==14.2.0 rpds-py==0.28.0 safetensors==0.6.2 Send2Trash==1.8.3 \
  sentry-sdk==2.43.0 setuptools==80.9.0 simplejson==3.20.2 six==1.17.0 smmap==5.0.2 sniffio==1.3.1 \
  soundfile==0.13.1 soupsieve==2.8 stack-data==0.6.3 sympy==1.14.0 terminado==0.18.1 \
  tinycss2==1.4.0 tokenizers==0.22.1 tomli==2.3.0 toolz==1.1.0 tornado==6.5.2 tqdm==4.67.1 \
  traitlets==5.14.3 transformers==4.57.1 typeguard==2.13.3 types-python-dateutil==2.9.0.20251008 \
  typing-inspection==0.4.2 typing_extensions==4.15.0 tzdata==2025.2 uri-template==1.3.0 urllib3==2.5.0 \
  wadler_lindig==0.1.7 wandb==0.22.3 wcwidth==0.2.14 webcolors==25.10.0 webencodings==0.5.1 \
  websocket-client==1.9.0 wheel==0.45.1 widgetsnbextension==4.0.15 xyzservices==2025.10.0 zipp==3.23.0
pip install --no-deps equinox==0.13.2
pip install --no-deps jaxtyping==0.2.33
pip install --no-deps treescope==0.1.10
python - << 'PY'
import torch, jax
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("jax:", jax.__version__, "devices:", jax.devices())
PY
pip install --no-deps torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge cuda-nvcc=12.1 -y



conda remove --all -n torchjax

#run before each launch
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
hash -r
export PYTHONPATH="/home/coder/project:$PYTHONPATH"

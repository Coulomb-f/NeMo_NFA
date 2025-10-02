# 1. Start with a modern PyTorch container
FROM nvcr.io/nvidia/pytorch:25.09-py3

# 2. Install essential system packages, including git and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /workspace

# 4. Initialize a Git repo and forcibly fetch your corrected branch.
RUN git init && \
    git remote add origin https://github.com/Coulomb-f/NeMo_NFA.git && \
    git fetch --depth 1 origin stable-v2.5.0 && \
    git reset --hard FETCH_HEAD

# 5. Install dependencies (with mcore enabled).
# RUN bash docker/common/install_dep.sh --library trtllm --mode install
RUN bash docker/common/install_dep.sh --library te --mode install
# RUN bash docker/common/install_dep.sh --library mcore --mode install
RUN bash docker/common/install_dep.sh --library vllm --mode install
RUN bash docker/common/install_dep.sh --library extra --mode install

# 6. Install the NeMo toolkit itself
RUN pip install --verbose ".[all]"

# 7. Set a default command to open a shell
CMD ["bash"]
# DistributedTraining
This repository contains code for distributed training across multiple tasks.  
All examples have been tested on a single node equipped with 4×NVIDIA A100 GPUs (40 GB each).

# Run This Code

```python
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=10001 ddplearning.py --dist-backend nccl --epochs 100 --batch-size 2048
```


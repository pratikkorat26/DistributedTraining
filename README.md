# DistributedTraining
I will write code for distributed training for various task in this repository

# Run This Code

```python
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=10001 ddplearning.py --dist-backend nccl --epochs 100 --batch-size 2048
```

# FedSDC
The FedSDC project simulates a distributed environment based on the [Flower](https://github.com/adap/flower) framework and can be run using the following command:

```
RAY_USE_MULTIPROCESSING_CPU_COUNT=32 CUDA_VISIBLE_DEVICES=1,2 flwr run . local-sim-gpu
```

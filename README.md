# Edge-Computing-Assignment-3
Assignment 3 of edge computing. Abyaz Israr 22-2056 DS-9A

This is my implementation of model optimization on a ResNet-18 trained on CIFAR-10.
It covers quantization (PTQ and QAT), pruning (unstructured and structured), and
ONNX export for CPU deployment. Everything was built and run in Google Colab.

## What you need

Python 3.10 or newer, and these packages:

  torch, torchvision
  torch-pruning
  onnx, onnxruntime, onnxscript
  numpy, pandas

To install them:

  pip install torch torchvision torch-pruning onnx onnxruntime onnxscript numpy pandas

In Colab most of these are already there, the notebook installs the rest when it runs.

## How to run

Open the notebook in Colab and turn on the GPU first
(Runtime > Change runtime type > T4 GPU). Then just run the cells top to bottom:

  Cells 1 to 9    train the FP32 baseline, then do PTQ and QAT, then benchmark them
  Cells 10 to 14  do the unstructured and structured pruning, then benchmark them
  Cells 15 to 16  export everything to ONNX and test it on CPU with ONNX Runtime

Run them in order. The later cells use variables from the earlier ones, so don't skip around.
The baseline training is the slowest part. On a T4 it takes a few minutes.

## Models you get

  FP32 baseline        q1_quantization/checkpoints/fp32_baseline.pth
  PTQ INT8             q1_quantization/checkpoints/ptq_int8.pt
  QAT INT8             q1_quantization/checkpoints/qat_int8.pt
  Unstructured pruned  q2_pruning/checkpoints/unstructured_pruned.pth
  Structured pruned    q2_pruning/checkpoints/structured_pruned.pth
  ONNX versions        onnx_models/

## Where the results are

The benchmark numbers and predictions get saved as CSV files:

  q1_quantization/results/   quantization accuracy, size, latency and predictions
  q2_pruning/results/        pruning accuracy, params, sparsity, size, latency
  onnx_models/               ONNX benchmark results

## A few things worth knowing

The INT8 quantized PyTorch models only run on CPU, that is just how PyTorch
quantization works (fbgemm backend). So all the fair speed and accuracy comparisons
are done on CPU, which also makes sense since edge devices are usually CPU based.

PyTorch's INT8 models don't export to ONNX cleanly, so for the ONNX INT8 model I used
ONNX Runtime's own static quantization instead. The float models (baseline and both
pruned ones) export to ONNX normally.

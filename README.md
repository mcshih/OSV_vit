# Transformer for OSV

## Commend
### testing
```python
CUDA_VISIBLE_DEVICES=0 python demo_vit.py --data ./../BHSig260/Bengali --name B_1 --batchSize 1 --imageSize 172 --model_type olp --test_only --load "model_weight.pt"
```

data: dataset path
name: model path (./saved_models/B_1/model_weight.pt)

### test for single pair

[single_test_vit.py#L143](https://github.com/mcshih/OSV_vit/blob/main/single_test_vit.py#L143): change to corresponding image paths

```python
CUDA_VISIBLE_DEVICES=0 python single_test_vit.py --name B_1 --model_type olp --imageSize 172 --load "model_weight.pt"

```
# Baseline without quantization
python main_loso.py --no-quant
# Post-training quantization: Weight & bias are int8
python quantize_loso --no-quant

# Softsign quantization, shared axis. Weight & bias are float 32
python main_loso.py 
# Post-training quantization: Weight & bias are int8
python quantize_loso.py

# Softsign quantization, per axis
python main_loso.py --per-channel-quant
# Post-training quantization: Weight & bias are int8
python quantize_loso.py --per-channel-quant

# Export model to tflite to verify inference time
python export.py
```
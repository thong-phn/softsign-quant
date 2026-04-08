UCI HAR
```
# Baseline without quantization
python main_loso.py --quantization 'no'
# Post-training quantization: Weight & activation are int8
python quantize_loso --no-quant

# Softsign quantization, shared axis. Weight & activation are float 32
python main_loso.py 
# Post-training quantization: Weight & activation are int8
python quantize_loso.py

# Softsign quantization, per axis
python main_loso.py --per-channel-quant
# Post-training quantization: Weight & activation are int8
python quantize_loso.py --per-channel-quant

# Export model to tflite to verify inference time
python export.py
```

WEAR

# Softsign quantization, shared axis. Weight & activation are float 32
python wear_main_loso.py --quantization 'softsign' --run_name 'updateSSFunction'
python wear_quantize_loso.py --quantization 'softsign' # Quantized W8A8

python wear_main_loso.py --quantization 'softsign' --per-channel --run_name 'updateSSFunction'
python wear_quantize_loso.py --quantization 'softsign' # Quantized W8A8
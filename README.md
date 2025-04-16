
Command to run multimodal model:
```
docker run -e MM_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct -e HF_TOKEN=hf_token --gpus all image_name both
```

Command to run text and image model:
```
docker run -e TEXT_MODEL__MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct -e TEXT_MODEL__TEXT_CLASSIFIER=in_context -e IMAGE_MODEL__MODEL_PATH=/app/image_model -v path_to_image_checkpoint:/app/image_model -e HF_TOKEN=hf_token --gpus all image_name both
```

Command to run text model
```
docker run -e MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct -e TEXT_CLASSIFIER=in_context -e HF_TOKEN=hf_token --gpus all image_name text
```






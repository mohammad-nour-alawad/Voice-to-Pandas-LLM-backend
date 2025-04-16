CUDA_VISIBLE_DEVICES=2 uvicorn api:app --host 0.0.0.0 --port 6000

- for espeak backend:

sudo apt-get update
sudo apt-get install -y espeak-ng
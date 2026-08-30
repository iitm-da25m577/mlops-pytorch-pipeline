# mlops-pytorch-pipeline

**Roll Number:** DA25M577

A Fashion-MNIST image classifier taken from local development through Docker
containerization to a full Kubernetes training + serving deployment.

## Architecture

```
                 ┌─────────────────────┐
                 │   configs/           │
                 │  training_config.yaml│
                 └──────────┬───────────┘
                            │
   ┌────────────────────────▼─────────────────────────┐
   │ Docker: mlops-train:DA25M577  (docker/Dockerfile.train) │
   │   src/train.py -> model.py + dataset.py            │
   │   reads config, trains, early-stops, checkpoints   │
   └────────────────────────┬─────────────────────────┘
                            │ writes checkpoint
                            ▼
                 ┌──────────────────────┐
                 │ checkpoints volume /  │
                 │ PVC (checkpoints-pvc) │
                 └──────────┬───────────┘
                            │ read-only mount
   ┌────────────────────────▼─────────────────────────┐
   │ Docker: mlops-serve:DA25M577  (docker/Dockerfile.serve) │
   │   src/serve.py (FastAPI) -> GET /health            │
   │                          -> POST /predict          │
   └────────────────────────┬─────────────────────────┘
                            │
                            ▼
        Kubernetes: Deployment (2 replicas) + Service (ClusterIP:80)
                       + HPA (CPU-based, 2-5 replicas)
```

On Kubernetes: `k8s/namespace.yaml` creates the `ml-training` namespace,
`k8s/configmap.yaml` supplies `training_config.yaml`, `k8s/training-job.yaml`
runs a one-shot training `Job` backed by PVCs for `/app/data` and
`/app/checkpoints`, and `k8s/serving-deployment.yaml` +
`k8s/serving-service.yaml` + `k8s/hpa.yaml` serve predictions from the
resulting checkpoint.

## Local setup

**Requires Python 3.11** (torch==2.3.1 has no wheels for newer Python versions like 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements/train.txt
python src/train.py --config configs/training_config.yaml
python src/train.py --config configs/training_config.yaml
```

## Docker

```bash
# Train
docker build -f docker/Dockerfile.train -t mlops-train:DA25M577 .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/checkpoints:/app/checkpoints mlops-train:DA25M577

# Serve
docker build -f docker/Dockerfile.serve -t mlops-serve:DA25M577 .
docker run --rm -p 8080:8080 -v ${PWD}/checkpoints:/app/checkpoints mlops-serve:DA25M577

# Test
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
curl http://localhost:8080/health
```

## Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=600s

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Project structure

See `k8s/`, `docker/`, `src/`, `configs/`, `requirements/`, `tests/`.

## Git workflow

- `main` — production-ready, only updated via merged PRs
- `develop` — integration branch
- `feature/*` — one branch per unit of work, merged into `develop` via PR,
  then `develop` merged into `main`
# mlops-pytorch-pipeline

#!/usr/bin/env bash
set -e

# =========================================================
# PART 1: Initialize git in the current folder (works in-place,
# e.g. on a downloaded/extracted zip with no .git yet)
# NOTE: Before running this, manually:
#   1. Delete the old GitHub repo (Settings -> Danger Zone -> Delete)
#   2. Create a new EMPTY repo named mlops-pytorch-pipeline (no README, no template)
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for f in src configs docker k8s requirements tests README.md .gitignore test_image.png; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected '$f' not found in $SCRIPT_DIR — is this the right folder?"
    exit 1
  fi
done

if [ ! -d .git ]; then
  git init
  git branch -M main
else
  echo "Already a git repo — reusing existing .git"
fi

git add .
git commit -m "chore: initial project structure with model, docker, and k8s manifests" || echo "Nothing new to commit."

if ! git remote get-url origin &>/dev/null; then
  git remote add origin https://github.com/iitm-da25m577/mlops-pytorch-pipeline.git
else
  git remote set-url origin https://github.com/iitm-da25m577/mlops-pytorch-pipeline.git
fi
git push -u origin main

if git show-ref --verify --quiet refs/heads/develop; then
  git checkout develop
else
  git checkout -b develop
fi
git push -u origin develop

# =========================================================
# PART 2: GPU prerequisites (NVIDIA GPU detected -> enabled)
# =========================================================
if ! command -v minikube &> /dev/null; then
  echo "minikube not found. Installing..."
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  sudo install minikube-linux-amd64 /usr/local/bin/minikube
  rm -f minikube-linux-amd64
fi

if ! command -v kubectl &> /dev/null; then
  echo "kubectl not found. Installing..."
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  sudo install kubectl /usr/local/bin/kubectl
  rm -f kubectl
fi

if ! command -v nvidia-container-toolkit &> /dev/null; then
  sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

if minikube status 2>/dev/null | grep -q "Running"; then
  echo "minikube already running — restarting with GPU support to be safe."
fi
minikube delete --purge || true
minikube start --driver=docker --container-runtime=docker --gpus=all
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
echo "Waiting for GPU to register as allocatable..."
for i in {1..30}; do
  GPU_CHECK=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable}' | grep -o "nvidia.com/gpu" || true)
  if [ -n "$GPU_CHECK" ]; then
    echo "GPU is allocatable."
    break
  fi
  sleep 5
done
if [ -z "$GPU_CHECK" ]; then
  echo "WARNING: GPU did not register after 150s. Falling back to CPU-only training-job.yaml."
  sed -i '/nodeSelector:/,+6d' k8s/training-job.yaml
  sed -i '/nvidia.com\/gpu: 1/d' k8s/training-job.yaml
fi

# =========================================================
# PART 3: Build images inside minikube's docker context
# =========================================================
eval $(minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:DA25M577 .
docker build -f docker/Dockerfile.serve -t mlops-serve:DA25M577 .
eval $(minikube docker-env -u)

# =========================================================
# PART 4: Deploy training + serving on Kubernetes
# =========================================================
kubectl delete namespace ml-training --ignore-not-found
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

echo "Waiting for training job to complete (up to 30 min)..."
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=1800s

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

echo "Waiting for serving deployment to become available..."
kubectl wait --for=condition=available deployment/model-serving -n ml-training --timeout=120s

# =========================================================
# PART 5: Validate end-to-end and save the log
# =========================================================
pkill -f "kubectl port-forward" || true
sleep 1

find_free_port() {
  for p in 8081 8082 8083 8084 8090 9090; do
    if ! (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null; then
      echo "$p"
      return
    fi
  done
  echo "0"
}

FREE_PORT=$(find_free_port)
if [ "$FREE_PORT" == "0" ]; then
  echo "ERROR: no free port found among candidates. Kill stray processes and rerun."
  exit 1
fi
echo "Using local port $FREE_PORT for port-forward."

{
  kubectl get pods -n ml-training
  kubectl describe deployment model-serving -n ml-training
  (kubectl port-forward svc/model-serving ${FREE_PORT}:80 -n ml-training &)
  sleep 5
  curl -sf http://localhost:${FREE_PORT}/health || echo "health check failed"
  echo
  curl -sf -X POST http://localhost:${FREE_PORT}/predict -F "image=@test_image.png" || echo "predict call failed"
  echo
} | tee validation-log.txt

pkill -f "kubectl port-forward" || true

# =========================================================
# PART 6: Commit validation log
# =========================================================
git add validation-log.txt
git commit -m "docs: add end-to-end validation logs" || true
git push -u origin develop

echo "DONE. Now open PRs manually on GitHub (or via 'gh pr create') for develop -> main, then submit repo + PR links."
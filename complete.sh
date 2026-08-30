#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO="iitm-da25m577/mlops-pytorch-pipeline"

echo "=== Working directory: $SCRIPT_DIR ==="

# =========================================================
# PART 0: Sanity check — required project files present
# =========================================================
for f in src configs docker k8s requirements tests README.md .gitignore test_image.png; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected '$f' not found in $SCRIPT_DIR — is this the right folder?"
    exit 1
  fi
done

# =========================================================
# PART 1: Git repo — init in place, push main + develop
# NOTE: before first run, manually delete any old/contaminated
# GitHub repo and create a fresh EMPTY one with this name.
# =========================================================
if [ ! -d .git ]; then
  git init
  git branch -M main
else
  echo "Already a git repo — reusing existing .git"
fi

git add .
git commit -m "chore: initial project structure with model, docker, and k8s manifests" || echo "Nothing new to commit."

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "https://github.com/${REPO}.git"
else
  git remote set-url origin "https://github.com/${REPO}.git"
fi
git push -u origin main

if git show-ref --verify --quiet refs/heads/develop; then
  git checkout develop
elif git show-ref --verify --quiet refs/remotes/origin/develop; then
  git checkout -b develop origin/develop
else
  git checkout -b develop
fi
git push -u origin develop

# =========================================================
# PART 2: Tooling — minikube, kubectl, GPU stack (with fallback)
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

if ! command -v gh &> /dev/null; then
  echo "GitHub CLI (gh) not found. Installing..."
  type -p curl >/dev/null || sudo apt install curl -y
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
  sudo apt update
  sudo apt install gh -y
fi
if ! gh auth status &> /dev/null; then
  echo "Not logged into gh. Launching login flow..."
  gh auth login
fi

GPU_AVAILABLE=0
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected — enabling GPU scheduling."
  if ! command -v nvidia-container-toolkit &> /dev/null; then
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  minikube delete --purge || true
  minikube start --driver=docker --container-runtime=docker --gpus=all
  kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml \
    || echo "Device plugin already present — continuing."

  echo "Waiting for GPU to register as allocatable..."
  GPU_CHECK=""
  for i in {1..30}; do
    GPU_CHECK=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable}' | grep -o "nvidia.com/gpu" || true)
    if [ -n "$GPU_CHECK" ]; then
      echo "GPU is allocatable."
      GPU_AVAILABLE=1
      break
    fi
    sleep 5
  done
  if [ "$GPU_AVAILABLE" == "0" ]; then
    echo "WARNING: GPU did not register after 150s. Falling back to CPU-only training-job.yaml."
    sed -i '/nodeSelector:/,+6d' k8s/training-job.yaml
    sed -i '/nvidia.com\/gpu: 1/d' k8s/training-job.yaml
  fi
else
  echo "No NVIDIA GPU detected — using CPU-only path."
  if minikube status &>/dev/null; then
    echo "minikube already running — reusing existing cluster."
  else
    minikube start --driver=docker
  fi
  sed -i '/nodeSelector:/,+6d' k8s/training-job.yaml 2>/dev/null || true
  sed -i '/nvidia.com\/gpu: 1/d' k8s/training-job.yaml 2>/dev/null || true
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
# PART 6: Commit validation log on a feature branch, PR it into develop
# =========================================================
CURRENT_BRANCH=$(git branch --show-current)
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes detected on '$CURRENT_BRANCH' — stashing so nothing is lost."
  git stash push -u -m "auto-stash before PR flow"
  STASHED=1
else
  STASHED=0
fi

git checkout develop
if ! git pull origin develop; then
  echo "ERROR: 'git pull origin develop' failed — local/remote history likely diverged."
  echo "Resolve manually, then re-run this script."
  exit 1
fi
if [ "$STASHED" == "1" ] && [ "$CURRENT_BRANCH" == "develop" ]; then
  git stash pop || echo "WARNING: stash pop had conflicts — resolve manually."
fi

BRANCH="feature/e2e-validation-$(date +%s)"
git checkout -b "$BRANCH"
git add validation-log.txt k8s/training-job.yaml
if git diff --cached --quiet; then
  echo "No new changes to commit — skipping this PR."
  git checkout develop
else
  git commit -m "docs: add end-to-end validation logs"
  git push -u origin "$BRANCH"
  gh pr create --repo "$REPO" --base develop --head "$BRANCH" \
    --title "docs: add end-to-end validation logs" \
    --body-file validation-log.txt
  gh pr merge --repo "$REPO" "$BRANCH" --merge --delete-branch
  git checkout develop
  git pull origin develop
fi

# =========================================================
# PART 7: Final PR — develop -> main
# =========================================================
gh pr create --repo "$REPO" --base main --head develop \
  --title "Final submission: MLOps PyTorch pipeline" \
  --body-file validation-log.txt || echo "PR may already exist — check 'gh pr list'."

gh pr merge --repo "$REPO" develop --merge || echo "Merge may need manual confirmation — check on GitHub."

echo "=== DONE ==="
echo "Repo: https://github.com/${REPO}"
echo "Submit this repo link and the final merged PR link on the course platform."
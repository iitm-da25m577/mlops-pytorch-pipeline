# Reflection

This project took a Fashion-MNIST classifier from local training through
Docker containerization to a Kubernetes training + serving deployment
(namespace, ConfigMap, Job with PVCs, Deployment with health probes, Service,
and an HPA).

## What was hardest

The most time-consuming part wasn't the PyTorch model itself but the
environment setup around it — a lot of boilerplate before anything actually
ran. Two concrete examples: minikube/kubectl weren't installed initially, and
a stray `docker run -p 8080:8080` container from local testing kept blocking
the same port during `kubectl port-forward` later, which took a few rounds of
`lsof` and `docker ps` (across both the host and minikube's internal Docker
context) to track down. Separately, downloaded `kubectl` and
`minikube-linux-amd64` binaries got committed by accident, which GitHub
rejected for exceeding its file size limit — fixed with `git reset --soft`
back to `develop` and a proper `.gitignore` entry before recommitting.

On the modeling side, [ADD YOUR OWN DETAIL HERE: what you observed — e.g.
"the CNN's ReLU activations occasionally stalled during early training,
which is a known dead-neuron risk with ReLU when large gradients push a
unit's output permanently negative; I experimented with swapping in Tanh in
the conv blocks to see if it helped."]

## What I'd do differently

Reduce boilerplate by leaning on an existing framework instead of hand-rolled
training loops and config wiring — e.g. PyTorch Lightning to cut down the
train/eval loop code, and Hydra or Optuna to manage and automate
hyperparameter sweeps (learning rate, batch size, architecture choice)
instead of a single static YAML. This would also make the early-stopping and
checkpointing logic reusable across experiments rather than rewritten per
project.

## What I learned

[ADD 2-3 SENTENCES: your own takeaway — e.g. about debugging distributed
systems, reading Kubernetes events, or something about model behavior]
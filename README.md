# LLM Cost Optimizer — Infrastructure PoC

## Overview

This repository contains the **cloud infrastructure and deployment setup** for the initial Proof of Concept (PoC) of an **LLM Cost Optimization / Distillation as a Service (DaaS)** solution.

The objective of this PoC is to validate whether a lightweight, task-specific student model can perform **ticket summarization** with acceptable quality while significantly reducing inference cost compared to a larger proprietary LLM.

This repository focuses on the **infrastructure, networking, security, containerization, deployment, and automation** required to run and serve the student model.

> **Current Phase:** Infrastructure & Student Model PoC
> **Primary Use Case:** Ticket Summarization
> **Cloud Provider:** AWS
> **Student Model:** Qwen 1.7B
> **Teacher Model:** Claude Sonnet — planned for a later phase

---

## Architecture

The initial architecture is designed to keep the PoC simple, secure, and cost-conscious.

```text
                         ┌───────────────────┐
                         │   Users / Clients │
                         └─────────┬─────────┘
                                   │
                                HTTPS
                                   │
                                   ▼
                     ┌────────────────────────┐
                     │    HAProxy EC2         │
                     │       t3.micro         │
                     │                        │
                     │  Reverse Proxy         │
                     │  Load Balancer         │
                     │  Bastion Host           │
                     └───────────┬────────────┘
                                 │
                          Private Traffic
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │     GPU EC2             │
                     │     g6.2xlarge           │
                     │                          │
                     │  Private Subnet         │
                     │                          │
                     │  ┌────────────────────┐  │
                     │  │ Docker Container   │  │
                     │  │                    │  │
                     │  │ FastAPI            │  │
                     │  │       ↓            │  │
                     │  │ Qwen 1.7B          │  │
                     │  │       ↓            │  │
                     │  │ Ticket Summary      │  │
                     │  └────────────────────┘  │
                     └───────────┬────────────┘
                                 │
                           Outbound Traffic
                                 │
                                 ▼
                         ┌──────────────┐
                         │ NAT Gateway  │
                         └──────┬───────┘
                                │
                                ▼
                             Internet
```

### Network Design

The infrastructure uses an AWS VPC with separate public and private subnets.

| Component   | Subnet  | Purpose                                        |
| ----------- | ------- | ---------------------------------------------- |
| HAProxy EC2 | Public  | External access, reverse proxy and SSH bastion |
| GPU EC2     | Private | Model training/inference                       |
| NAT Gateway | Public  | Outbound internet access for private resources |

The GPU instance is not directly exposed to the public internet.

---

## Infrastructure Components

### 1. AWS VPC

A dedicated VPC provides network isolation for the PoC.

Example:

```text
VPC: 10.0.0.0/16
```

Subnets:

```text
Public Subnet
10.0.1.0/24

Private Subnet
10.0.2.0/24
```

---

### 2. HAProxy EC2

A lightweight EC2 instance is used for the external reverse-proxy/load-balancing layer.

**Initial instance:**

```text
Instance Type: t3.micro
Location: Public Subnet
```

Responsibilities:

* Accept external HTTPS traffic
* Reverse proxy requests to the GPU server
* Provide a controlled entry point to the application
* Act as a bastion host for administrative SSH access during the PoC

---

### 3. GPU EC2

The GPU instance hosts the model training and inference environment.

**Recommended PoC configuration:**

```text
Instance Type: g6.2xlarge
GPU: NVIDIA L4
GPU Memory: 24 GB
vCPU: 8
RAM: 32 GiB
```

Responsibilities:

* Model training
* QLoRA fine-tuning
* Model evaluation
* Docker-based model serving
* FastAPI inference service

The GPU instance will remain in the **private subnet**.

---

### 4. Docker

The student model inference environment will be containerized using Docker.

Expected container structure:

```text
Docker Container
│
├── FastAPI
│
├── Model Serving
│
└── Qwen 1.7B
```

Containerization provides:

* Reproducible environments
* Dependency isolation
* Easier deployment
* Easier migration between environments
* Simplified future CI/CD integration

---

### 5. FastAPI

FastAPI will provide the HTTP API for model inference.

Example endpoint:

```http
POST /summarize
```

Example request:

```json
{
  "ticket": "Customer is unable to login to the application..."
}
```

Example response:

```json
{
  "summary": "Customer is unable to log in to the application and requires account access assistance."
}
```

---

## Security Design

Security is an important part of the infrastructure design.

### HAProxy Security Group

Expected inbound rules:

| Port | Source           | Purpose                   |
| ---- | ---------------- | ------------------------- |
| 443  | Internet         | HTTPS application traffic |
| 80   | Internet         | HTTP → HTTPS redirect     |
| 22   | Administrator IP | SSH administration        |

All other inbound traffic is denied.

### GPU EC2 Security Group

Expected inbound rules:

| Port | Source                           | Purpose                     |
| ---- | -------------------------------- | --------------------------- |
| 8000 | HAProxy Security Group           | FastAPI application traffic |
| 22   | Controlled administrative access | SSH                         |

The model API is **not directly exposed to the internet**.

Traffic follows:

```text
Internet
    │
    ▼
HAProxy
    │
    │ Port 8000
    ▼
GPU EC2
```

This prevents direct public access to the model server.

---

## SSH Access

During the PoC, the HAProxy EC2 instance can also act as a bastion host.

```text
Administrator
      │
      │ SSH
      ▼
HAProxy EC2
      │
      │ SSH / Private IP
      ▼
GPU EC2
```

This allows administration of the private GPU instance without exposing it directly to the internet.

A future implementation may use **AWS Systems Manager (SSM)** to reduce the need for SSH access.

---

## Outbound Internet Access

The GPU EC2 instance is placed in a private subnet but may require outbound internet connectivity for:

* Installing system packages
* Installing Python dependencies
* Pulling container images
* Downloading model files
* Accessing GitHub
* Accessing Hugging Face
* Calling external APIs when required

Outbound traffic can be routed through:

```text
Private GPU EC2
       ↓
NAT Gateway
       ↓
Internet Gateway
       ↓
Internet
```

No unsolicited inbound internet traffic is allowed directly to the GPU instance.

---

# Model Workflow

The overall model workflow is divided into separate phases.

## Current Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
QLoRA Fine-Tuning
   ↓
Qwen 1.7B
   ↓
Evaluation
   ↓
Dockerization
   ↓
FastAPI
   ↓
HAProxy
   ↓
Client
```

---

## Future Teacher Model Workflow

The Claude Sonnet teacher pipeline is intentionally **not part of the initial infrastructure implementation**.

It will be introduced after the student-model PoC demonstrates acceptable results.

Future workflow:

```text
Raw / Synthetic Ticket Data
          ↓
     Claude Sonnet
     Teacher Model
          ↓
   Generated Summaries
          ↓
      Validation
          ↓
    Training Dataset
          ↓
      QLoRA Training
          ↓
      Qwen 1.7B
```

This separation allows the team to first validate the core technical feasibility before investing in automated teacher-model data generation.

---

# GitHub and Version Control

GitHub is used for source-code and infrastructure version control.

The repository will contain:

```text
├── application/
├── docker/
├── ansible/
├── infrastructure/
├── scripts/
├── configs/
└── README.md
```

Large datasets, model checkpoints, and other large generated artifacts should not unnecessarily be committed directly to Git.

As the project grows, object storage such as Amazon S3 can be introduced for:

* Large datasets
* Teacher-generated datasets
* Model checkpoints
* Trained model artifacts
* Evaluation results

---

# Infrastructure Automation

Ansible will be used to automate configuration and deployment of the EC2 infrastructure.

Planned responsibilities include:

```text
Ansible
   │
   ├── Configure EC2
   ├── Install Docker
   ├── Configure NVIDIA environment
   ├── Install dependencies
   ├── Deploy application
   ├── Build / pull Docker image
   ├── Start containers
   └── Perform health checks
```

Example structure:

```text
ansible/
│
├── inventory/
│   └── hosts
│
├── playbooks/
│   ├── site.yml
│   ├── docker.yml
│   ├── gpu.yml
│   └── deploy.yml
│
└── roles/
    ├── docker/
    ├── gpu/
    └── application/
```

---

# CI/CD — Planned

CI/CD automation will be introduced after the initial deployment is stable.

Potential workflow:

```text
Developer
    │
    ▼
GitHub
    │
    ▼
GitHub Actions
    │
    ├── Test
    ├── Build Docker Image
    ├── Push Image
    └── Trigger Deployment
                    │
                    ▼
                 EC2
                    │
                    ▼
              New Container
```

For the initial PoC, expensive GPU training should **not automatically trigger on every GitHub change**.

Training automation will be introduced later with appropriate conditions and manual approval where necessary.

---

# Project Phases

## Phase 1 — Infrastructure Setup

* [ ] Create AWS VPC
* [ ] Create public subnet
* [ ] Create private subnet
* [ ] Configure route tables
* [ ] Configure Internet Gateway
* [ ] Configure NAT Gateway
* [ ] Create Security Groups
* [ ] Launch HAProxy EC2
* [ ] Launch GPU EC2
* [ ] Configure SSH access

---

## Phase 2 — Server Configuration

* [ ] Configure HAProxy
* [ ] Configure GPU environment
* [ ] Install NVIDIA drivers
* [ ] Install Docker
* [ ] Configure NVIDIA Container Toolkit
* [ ] Install Python/dependencies
* [ ] Configure Ansible automation

---

## Phase 3 — Model Deployment

* [ ] Receive trained student model
* [ ] Create Docker image
* [ ] Deploy Qwen 1.7B
* [ ] Implement FastAPI service
* [ ] Test local inference
* [ ] Deploy to GPU EC2
* [ ] Configure HAProxy routing

---

## Phase 4 — Evaluation

Measure:

### Model Quality

* ROUGE
* BERTScore
* Factuality
* Hallucination rate
* Human evaluation

### Infrastructure Performance

* Response latency
* Time to first token
* Tokens/second
* GPU utilization
* GPU memory usage
* CPU utilization
* RAM utilization

### Cost

* Cost per inference
* Cost per 1,000 tickets
* GPU infrastructure cost
* Teacher-model data-generation cost
* Overall cost comparison

The primary objective is to determine whether the smaller student model can provide **acceptable ticket-summarization quality at substantially lower inference cost**.

---

# Phase 5 — Teacher Model Automation

**Planned — Not part of the initial PoC**

* [ ] Integrate Claude Sonnet API
* [ ] Generate teacher summaries
* [ ] Validate generated outputs
* [ ] Build distillation dataset
* [ ] Automate dataset generation
* [ ] Trigger training pipeline
* [ ] Evaluate new student model
* [ ] Compare model versions

---

# Phase 6 — Production Readiness

If the PoC is successful and the solution moves toward an actual enterprise deployment:

* Replace single HAProxy instance with a highly available load-balancing architecture
* Introduce multiple inference instances
* Introduce proper model/artifact storage
* Implement monitoring and alerting
* Implement centralized logging
* Introduce model versioning
* Implement automated CI/CD
* Implement secure secrets management
* Implement autoscaling where appropriate
* Introduce stronger IAM controls
* Evaluate AWS managed load balancing
* Implement production-grade observability

---

# Repository Goals

The primary infrastructure goals are:

1. Build a secure AWS environment for the DaaS PoC.
2. Keep the infrastructure simple and cost-conscious.
3. Deploy the student model on GPU infrastructure.
4. Containerize the inference service.
5. Expose the service through a controlled reverse-proxy layer.
6. Automate infrastructure configuration using Ansible.
7. Establish a foundation for future CI/CD.
8. Keep the architecture extensible for future teacher-model and enterprise integrations.

---

# Current Architecture Summary

```text
                    CLIENT
                       │
                     HTTPS
                       │
                       ▼
              ┌─────────────────┐
              │ HAProxy EC2     │
              │ t3.micro        │
              │ Public Subnet   │
              └────────┬────────┘
                       │
                 Private Traffic
                       │
                       ▼
              ┌─────────────────┐
              │ GPU EC2         │
              │ g6.2xlarge      │
              │ Private Subnet  │
              │                 │
              │ Docker          │
              │   ↓             │
              │ FastAPI         │
              │   ↓             │
              │ Qwen 1.7B       │
              └────────┬────────┘
                       │
                 Outbound Only
                       │
                       ▼
                 NAT Gateway
                       │
                       ▼
                    Internet
```

---

## Team Responsibilities

### Infrastructure

Responsible for:

* AWS architecture
* VPC and networking
* Security Groups
* EC2 provisioning
* HAProxy
* Docker
* Ansible
* Deployment
* CI/CD
* Infrastructure security
* Monitoring and observability

### Model Development

Responsible for:

* Dataset preparation
* QLoRA fine-tuning
* Student model development
* Model evaluation
* Teacher model integration
* Distillation experiments
* Quality and cost analysis

---

## Status

**Project Stage: Initial Proof of Concept**

The current priority is to establish the infrastructure required to train, deploy, and serve the student model.

The Claude Sonnet teacher-model automation, automated retraining, and production-grade infrastructure will be implemented only after the initial PoC demonstrates sufficient technical and business feasibility.

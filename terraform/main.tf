# main.tf
# -------
# Provisions a single EC2 instance to run the FastAPI backend (api.py).
#
# Design choice: EC2, not Lambda. This app's dependencies (PyTorch,
# sentence-transformers, the embedding + reranker models) are large -
# well beyond AWS Lambda's deployment package limits, and the models
# benefit from staying loaded in memory between requests rather than
# a cold start on every invocation. A small EC2 instance is the
# honest, realistic choice here, not a workaround.
#
# What this provisions:
#   - One EC2 instance (t3.small - enough RAM for the embedding +
#     reranker models to run comfortably)
#   - A security group allowing SSH (22) and the API port (8000)
#   - user_data that installs dependencies and starts the API on boot
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-south-1" # Mumbai - closest region if deploying from India
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair, for SSH access"
  type        = string
  # No default on purpose - you must provide this (terraform.tfvars or -var flag)
}

variable "groq_api_key" {
  description = "Groq API key, injected into the instance as an environment variable"
  type        = string
  sensitive   = true
}

# Amazon Linux 2023 AMI, looked up dynamically so this doesn't go stale.
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_security_group" "rag_search_sg" {
  name        = "enterprise-rag-search-sg"
  description = "Allow SSH and API access for the RAG search backend"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten this to your own IP in a real deployment
  }

  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "rag_search_api" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.rag_search_sg.id]

  root_block_device {
    volume_size = 20 # GB - PyTorch + model weights need more than the 8GB default
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    groq_api_key = var.groq_api_key
  })

  tags = {
    Name = "enterprise-rag-search-api"
  }
}

output "instance_public_ip" {
  value       = aws_instance.rag_search_api.public_ip
  description = "Public IP of the API server - visit http://<this-ip>:8000/docs once boot finishes"
}

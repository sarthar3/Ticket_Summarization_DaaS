terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    # We need the TLS provider to generate the SSH key locally
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  # Hardcoding Mumbai region based on client location
  region = "ap-south-1" 

  default_tags {
    tags = {
      Project     = "AI-Ticket-Summarization"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}
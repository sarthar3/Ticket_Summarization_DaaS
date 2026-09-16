variable "environment" {
  description = "deployment environment name"
  type = string
  default = "dev"
}

variable "admin_ip" {
  description = "your personal/office public IP in CIDR format (e.g., 203.0.113.45/32) to allow SSH"
  type = string
}

variable "haproxy_instance_type" {
  description = "Instance type for HAProxy / Bastion host"
  type = string
  default = "t3.micro"
}

variable "inference_instance_type" {
  description = "Instance type for the inference server"
  type = string
  default = "c6a.2xlarge" 
}
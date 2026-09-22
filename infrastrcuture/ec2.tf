data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_instance" "haproxy" {
  ami = data.aws_ami.ubuntu.id
  instance_type = var.haproxy_instance_type
  subnet_id = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.haproxy_sg.id]
  key_name = aws_key_pair.generated_key.key_name # Links to auto-generated key
  associate_public_ip_address = true

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  tags = {
    Name = "HAProxy-Bastion"
  }
}

# Inference Server (Private Subnet - CPU Only)
resource "aws_instance" "inference_server" {
  ami                    = data.aws_ami.ubuntu.id  # Reusing the standard Ubuntu AMI
  instance_type          = var.inference_instance_type
  subnet_id              = aws_subnet.private.id
  vpc_security_group_ids = [aws_security_group.gpu_sg.id] # Keeping the same security group
  key_name               = aws_key_pair.generated_key.key_name

  root_block_device {
    volume_size = 50 # Can be smaller since you aren't pulling massive GPU containers
    volume_type = "gp3"
  }

  tags = {
    Name = "Qwen-CPU-Inference"
  }
}
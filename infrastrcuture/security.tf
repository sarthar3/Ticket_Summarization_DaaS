# HAProxy Security Group (Public)
resource "aws_security_group" "haproxy_sg" {
  name = "HAProxy-SG"
  description = "Allow inbound HTTPS/HTTP and admin SSH"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "HTTPS from Internet"
    from_port = 443
    to_port = 443
    protocol= "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP from Internet"
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH from Admin IP"
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = [var.admin_ip] # need to define this in variables.tf
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "HAProxy-SG"
  }
}

# GPU Server Security Group (Private)
resource "aws_security_group" "gpu_sg" {
  name = "GPU-SG"
  description = "Allow Port 8000 from HAProxy and SSH"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "FastAPI traffic from HAProxy"
    from_port = 8000
    to_port = 8000
    protocol = "tcp"
    security_groups = [aws_security_group.haproxy_sg.id]
  }

  ingress {
    description = "SSH from Bastion (HAProxy)"
    from_port = 22
    to_port = 22
    protocol = "tcp"
    security_groups = [aws_security_group.haproxy_sg.id]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "GPU-SG"
  }
}
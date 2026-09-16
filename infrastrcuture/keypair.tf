# Generates a secure private key
resource "tls_private_key" "ssh_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Registers the public key with AWS
resource "aws_key_pair" "generated_key" {
  key_name = "sparklehood-dev-key"
  public_key = tls_private_key.ssh_key.public_key_openssh
}

# Saves the private key locally
resource "local_file" "private_key_pem" {
  content = tls_private_key.ssh_key.private_key_pem
  filename = "${path.module}/sparklehood-dev-key.pem"
  file_permission = "0400"
}
output "haproxy_public_ip" {
  description = "Public IP address of HAProxy / Bastion host"
  value       = aws_instance.haproxy.public_ip
}

output "inference_server_private_ip" {
  description = "Private IP address of the Inference Server"
  value       = aws_instance.inference_server.private_ip
}

output "ssh_command_bastion" {
  description = "Command to SSH into HAProxy Bastion"
  value       = "ssh -i sparklehood-dev-key.pem ubuntu@${aws_instance.haproxy.public_ip}"
}

output "ssh_command_inference" {
  description = "Command to SSH into Inference server via Bastion ProxyJump"
  value       = "ssh -i sparklehood-dev-key.pem -J ubuntu@${aws_instance.haproxy.public_ip} ubuntu@${aws_instance.inference_server.private_ip}"
}
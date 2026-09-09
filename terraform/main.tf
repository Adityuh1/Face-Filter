# 1. AWS Provider Configuration
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
provider "aws" {
  region = "ap-south-1" # Mumbai region (change if your instance was created in us-east-1)
}

# 2. Security Group (Firewall)
resource "aws_security_group" "streamlit_tf_sg" {
  name        = "streamlit-tf-security-group"
  description = "Allow SSH and Streamlit web traffic"
  # Allow SSH (Port 22)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
    # Allow Streamlit Web (Port 8501)
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic (internet access for downloading packages)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "streamlit-tf-sg"
  }
}


# 3. EC2 Instance Provisioning
resource "aws_instance" "streamlit_tf_server" {
  ami           = "ami-00bb6a80f01f03502" # Ubuntu Server 24.04 / 26.04 LTS (ap-south-1)
  instance_type = "t3.small"
  key_name      = "SSH Key Pair" # Must match your downloaded SSH key name in AWS
  vpc_security_group_ids = [aws_security_group.streamlit_tf_sg.id]
  # 20 GB EBS Storage
  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = {
    Name = "FaceFilter-Terraform-Server"
  }
}

# 4. Output the Public IP automatically after creation
output "public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.streamlit_tf_server.public_ip
}
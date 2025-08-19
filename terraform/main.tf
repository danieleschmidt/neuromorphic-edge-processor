# Neuromorphic Edge Processor - Terraform Infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "neuromorphic-cluster"
}

variable "instance_type" {
  description = "EC2 instance type for worker nodes"
  type        = string
  default     = "c5.xlarge"
}

variable "min_nodes" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Node groups
  eks_managed_node_groups = {
    neuromorphic_nodes = {
      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.min_nodes
      
      instance_types = [var.instance_type]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        Workload    = "neuromorphic-computing"
      }
      
      taints = [
        {
          key    = "neuromorphic.ai/dedicated"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      update_config = {
        max_unavailable_percentage = 25
      }
    }
  }
  
  # aws-auth configmap
  manage_aws_auth_configmap = true
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
  }
}

# Security Groups
resource "aws_security_group" "neuromorphic_sg" {
  name_prefix = "${var.cluster_name}-neuromorphic-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "Neuromorphic API"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.cluster_name}-neuromorphic-sg"
    Environment = var.environment
  }
}

# ECR Repository
resource "aws_ecr_repository" "neuromorphic_repo" {
  name                 = "neuromorphic-edge-processor"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
  }
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "neuromorphic_policy" {
  repository = aws_ecr_repository.neuromorphic_repo.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 30 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 30
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "neuromorphic_artifacts" {
  bucket = "${var.cluster_name}-neuromorphic-artifacts-${random_string.suffix.result}"
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
  }
}

resource "aws_s3_bucket_versioning" "neuromorphic_artifacts_versioning" {
  bucket = aws_s3_bucket.neuromorphic_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "neuromorphic_artifacts_encryption" {
  bucket = aws_s3_bucket.neuromorphic_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# IAM Role for Kubernetes ServiceAccount
resource "aws_iam_role" "neuromorphic_service_role" {
  name = "${var.cluster_name}-neuromorphic-service-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:neuromorphic:neuromorphic-service-account"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
  }
}

# IAM Policy for S3 and ECR access
resource "aws_iam_policy" "neuromorphic_service_policy" {
  name        = "${var.cluster_name}-neuromorphic-service-policy"
  description = "IAM policy for neuromorphic edge processor service"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.neuromorphic_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.neuromorphic_artifacts.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "neuromorphic_service_policy_attachment" {
  role       = aws_iam_role.neuromorphic_service_role.name
  policy_arn = aws_iam_policy.neuromorphic_service_policy.arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "neuromorphic_logs" {
  name              = "/aws/eks/${var.cluster_name}/neuromorphic"
  retention_in_days = 30
  
  tags = {
    Environment = var.environment
    Project     = "neuromorphic-edge-processor"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.neuromorphic_repo.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket name for artifacts"
  value       = aws_s3_bucket.neuromorphic_artifacts.id
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}
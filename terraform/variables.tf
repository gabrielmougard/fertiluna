## Cloudflare credentials (set in terraform.tfvars — git-crypt encrypted) ##

variable "cloudflare_api_token" {
  description = "Cloudflare API token with Workers Scripts:Edit + the zone's DNS:Edit / Zone:Read"
  type        = string
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare account ID"
  type        = string
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for the domain (Overview page of the zone)"
  type        = string
}

## Deployment config ##

variable "cloudflare_environment" {
  description = "Deployment environment (staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["staging", "production"], var.cloudflare_environment)
    error_message = "Environment must be 'staging' or 'production'."
  }
}

variable "domain" {
  description = "Apex domain (e.g. fertiluna.com)"
  type        = string
}

variable "worker_name" {
  description = "Worker script name — MUST match `name` in wrangler.toml"
  type        = string
  default     = "fertiluna"
}

variable "enable_www_redirect" {
  description = "Create a www.<domain> -> apex 301 redirect rule"
  type        = bool
  default     = true
}

variable "enable_smart_placement" {
  description = "Enable Smart Placement for the worker"
  type        = bool
  default     = false
}

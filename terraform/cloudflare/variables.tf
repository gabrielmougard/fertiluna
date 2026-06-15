variable "cloudflare_environment" {
  description = "Deployment environment (staging, production)"
  type        = string
  default     = "production"
}

variable "cloudflare_account_id" {
  description = "Cloudflare account ID"
  type        = string
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for the domain"
  type        = string
}

variable "domain" {
  description = "Apex domain (e.g. fertiluna.com)"
  type        = string
}

variable "worker_name" {
  description = "Worker script name (must match `name` in wrangler.toml)"
  type        = string
  default     = "fertiluna"
}

variable "enable_www_redirect" {
  description = "Create a www -> apex 301 redirect rule"
  type        = bool
  default     = true
}

variable "enable_smart_placement" {
  description = "Enable Smart Placement for the worker"
  type        = bool
  default     = false
}

variable "compatibility_date" {
  description = "Worker compatibility date (match wrangler.toml)"
  type        = string
  default     = "2025-05-20"
}

variable "compatibility_flags" {
  description = "Worker compatibility flags (match wrangler.toml)"
  type        = list(string)
  default     = ["nodejs_compat"]
}

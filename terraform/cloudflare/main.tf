terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

# Provider is configured in the ROOT module and inherited here — no provider
# block in child modules (Terraform best practice).

locals {
  # Worker script name carries the environment so staging/prod don't collide,
  # while production keeps the bare name (must match wrangler.toml `name`).
  worker_name = var.cloudflare_environment == "production" ? var.worker_name : "${var.worker_name}-${var.cloudflare_environment}"
}

module "worker" {
  source = "./worker"

  account_id = var.cloudflare_account_id
  zone_id    = var.cloudflare_zone_id

  worker_name = local.worker_name
  domain      = var.domain

  enable_www_redirect    = var.enable_www_redirect
  enable_smart_placement = var.enable_smart_placement

  compatibility_date  = var.compatibility_date
  compatibility_flags = var.compatibility_flags
}

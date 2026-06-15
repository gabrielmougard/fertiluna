module "cloudflare" {
  source = "./cloudflare"

  cloudflare_account_id  = var.cloudflare_account_id
  cloudflare_zone_id     = var.cloudflare_zone_id
  cloudflare_environment = var.cloudflare_environment

  domain      = var.domain
  worker_name = var.worker_name

  enable_www_redirect    = var.enable_www_redirect
  enable_smart_placement = var.enable_smart_placement
}

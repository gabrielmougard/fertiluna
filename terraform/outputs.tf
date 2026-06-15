output "worker_url" {
  description = "Public URL of the deployed Cloudflare Worker"
  value       = module.cloudflare.worker_url
}

output "worker_name" {
  description = "Name of the Cloudflare Worker script (deploy target for wrangler)"
  value       = module.cloudflare.worker_name
}

output "custom_domain" {
  description = "Apex hostname bound to the worker"
  value       = module.cloudflare.custom_domain
}

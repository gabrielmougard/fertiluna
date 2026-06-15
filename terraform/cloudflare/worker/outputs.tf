output "worker_name" {
  description = "Name of the deployed worker"
  value       = cloudflare_workers_script.this.script_name
}

output "worker_id" {
  description = "ID of the deployed worker"
  value       = cloudflare_workers_script.this.id
}

output "custom_domain" {
  description = "Apex hostname bound to the worker"
  value       = cloudflare_workers_custom_domain.apex.hostname
}

output "url" {
  description = "Public URL of the worker"
  value       = "https://${var.domain}"
}

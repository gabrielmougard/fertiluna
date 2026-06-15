output "worker_url" {
  description = "Public URL of the deployed worker"
  value       = module.worker.url
}

output "worker_name" {
  description = "Name of the deployed worker script"
  value       = module.worker.worker_name
}

output "custom_domain" {
  description = "Apex hostname bound to the worker"
  value       = module.worker.custom_domain
}

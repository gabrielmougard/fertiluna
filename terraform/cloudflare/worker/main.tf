terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# WORKER SCRIPT (bootstrap placeholder)
# =============================================================================
# Terraform creates the Worker *named* `var.worker_name` so the custom domain
# can attach to it. The REAL code (the Astro SSR bundle + static assets) is
# shipped by `wrangler deploy` (locally or from CI) to this same script name.
# `ignore_changes` makes Terraform stop managing everything wrangler owns —
# code, assets, bindings (vars/secrets), compat settings — so the two tools
# never fight over the script. Terraform owns "the worker + domain exist";
# wrangler owns "what the worker runs".
resource "cloudflare_workers_script" "this" {
  account_id  = var.account_id
  script_name = var.worker_name

  content     = file("${path.module}/placeholder.js")
  main_module = "worker.js"

  compatibility_date  = var.compatibility_date
  compatibility_flags = var.compatibility_flags

  # NOTE: `observability` and `placement` are intentionally NOT managed here.
  # The cloudflare provider (v5.20.0) crashes reading back the observability
  # config (traces.propagation_policy schema drift). These are runtime concerns
  # owned by wrangler anyway — set them in wrangler.toml so they survive
  # `wrangler deploy` (which ignore_changes deliberately lets wrangler own).

  placement = var.enable_smart_placement ? { mode = "smart" } : null

  lifecycle {
    ignore_changes = [
      content,
      content_file,
      content_sha256,
      bindings,
      assets,
      main_module,
      compatibility_date,
      compatibility_flags,
      observability,
      placement,
    ]
  }
}

# =============================================================================
# CUSTOM DOMAIN — link fertiluna.com → the worker
# =============================================================================
# This is the Workers "Custom Domain" feature: Cloudflare provisions the apex
# DNS record AND the edge TLS certificate automatically. No dummy A/AAAA records
# and no Workers Route needed.
resource "cloudflare_workers_custom_domain" "apex" {
  account_id = var.account_id
  zone_id    = var.zone_id
  hostname   = var.domain
  # Bind by the known script NAME (var.worker_name) rather than a reference to
  # the script resource. This decouples the domain from the script resource so a
  # provider read-bug on the script (observability schema drift) can't block the
  # domain, and lets the script be owned outside TF state if needed. depends_on
  # still guarantees the script exists first on a fresh apply.
  service    = var.worker_name
  depends_on = [cloudflare_workers_script.this]
}

# =============================================================================
# WWW -> APEX redirect (301)
# =============================================================================
# We do NOT serve the worker on www as well (that would be duplicate content on
# a second hostname). Instead a single dynamic-redirect rule sends
# www.fertiluna.com/* -> https://fertiluna.com/* (301, query string preserved),
# consolidating SEO signals on the apex. Toggle with var.enable_www_redirect.
resource "cloudflare_ruleset" "www_redirect" {
  count = var.enable_www_redirect ? 1 : 0

  zone_id = var.zone_id
  name    = "${var.worker_name} www to apex redirect"
  kind    = "zone"
  phase   = "http_request_dynamic_redirect"

  rules = [{
    ref         = "www_to_apex"
    description = "Redirect www to the apex domain"
    expression  = "(http.host eq \"www.${var.domain}\")"
    action      = "redirect"
    enabled     = true
    action_parameters = {
      from_value = {
        status_code           = 301
        preserve_query_string = true
        target_url = {
          expression = "concat(\"https://${var.domain}\", http.request.uri.path)"
        }
      }
    }
  }]
}

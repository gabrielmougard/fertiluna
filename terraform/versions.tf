terraform {
  required_version = ">= 1.5"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }

  # Recommended: a remote backend so state (which contains the worker id + any
  # sensitive values) is not kept on a laptop. Example with Cloudflare R2 via the
  # S3-compatible API — uncomment and fill in once the bucket exists:
  #
  # backend "s3" {
  #   bucket                      = "fertiluna-tfstate"
  #   key                         = "cloudflare/terraform.tfstate"
  #   region                      = "auto"
  #   endpoints                   = { s3 = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com" }
  #   skip_credentials_validation = true
  #   skip_region_validation      = true
  #   skip_requesting_account_id  = true
  #   skip_s3_checksum            = true
  #   use_path_style              = true
  # }
}

# The provider is configured ONCE here in the root module; child modules inherit
# it (they only declare required_providers, never their own provider block).
provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

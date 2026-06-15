// Placeholder worker script
// Real deployment happens via wrangler
// This file exists only for Terraform to create the worker resource

export default {
  async fetch(request, env, ctx) {
    return new Response("Worker deployed via Terraform. Update via wrangler.", {
      headers: { "Content-Type": "text/plain" },
    });
  },
};

# Render Free Tier Demo Deployment Guide

This document outlines how to deploy the **Autonomous AI Software Engineer** API and control plane on **Render's Free Tier** using the dedicated `render.demo.yaml` blueprint.

---

## 1. Overview & Architectural Trade-offs

Render's free tier web services provide a zero-cost environment suitable for public demonstrations, portfolio reviews, and interactive evaluation.

### Free Tier Limitations & Ephemeral Storage
- **No Persistent Disks**: Render's free tier does not support persistent disks. Consequently, all SQLite databases (`workspace/checkpoints.db`, `workspace/telemetry.db`, `workspace/idempotency.db`) and workspace lock files reside on the container's local filesystem.
- **Sleep on Inactivity**: Free tier instances spin down after 15 minutes of inactivity. When a spin-down or redeployment occurs, the local container filesystem is reset to an empty state.
- **Session Ephemerality**: Multi-turn workflows, human-in-the-loop (HITL) pause/resumes, and idempotency checks function seamlessly **within the lifetime of the active container instance**. However, state does **not** survive container restarts.

### Preserved Production Safety Invariants
Even in Free Demo mode (`DEPLOYMENT_MODE=demo`), the system maintains strict security invariants:
- **Authentication & RBAC**: `AUTH_MODE=production` is enforced; unauthenticated API requests are rejected with HTTP 401.
- **Tenant Isolation**: Runs, checkpoints, idempotency keys, and audit logs remain strictly scoped by `organization_id`.
- **HITL Approval Gates**: Code commits and PR publication pause for human reviewer decision.
- **Policy Enforcement**: Pre-commit policy validation blocks forbidden files, excessive changes, and credential exposure.
- **Sandbox Security**: Automated testing executes with `shell=False` in an isolated environment with sensitive credentials stripped.
- **Safe LLM Execution**: 60-second bounded request timeouts and structured error classification prevent hangs.

---

## 2. Deploying on Render Free Tier

### Step 1: Connect Repository to Render
1. In the [Render Dashboard](https://dashboard.render.com/), select **New +** → **Blueprint**.
2. Select your repository: `agentic-ai-software-engineer`.
3. Switch the branch selector to: `demo/free-render`.
4. If prompted for the blueprint path, specify: `render.demo.yaml`.

### Step 2: Configure Environment Variables
Render will detect `render.demo.yaml` and prompt for sensitive environment variables marked with `sync: false`:

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` (or `openai`, `nvidia`) | Primary LLM provider |
| `GOOGLE_API_KEY` | *(your Gemini API key)* | Required if `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | *(your OpenAI API key)* | Required if `LLM_PROVIDER=openai` (or optional fallback) |
| `GITHUB_TOKEN` | *(your GitHub PAT with `repo` scope)* | Autonomous commit, branch, and PR operations |

*Note: `DEPLOYMENT_MODE=demo` and `AUTH_MODE=production` are pre-configured automatically by `render.demo.yaml`.*

### Step 3: Launch Deployment
1. Click **Apply**.
2. Monitor build logs: Render installs dependencies from `requirements.txt` and starts Uvicorn:
   ```bash
   uvicorn backend.api.app:app --host 0.0.0.0 --port $PORT
   ```
3. Once live, Render marks the service healthy via `/health`.

---

## 3. Post-Deployment Verification

### 1. Liveness & Storage Notice Check
```bash
curl -fsS https://<your-render-service>.onrender.com/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "agentic-ai-software-engineer-api",
  "version": "1.0.0",
  "deployment_mode": "demo",
  "storage_notice": "DEMO_EPHEMERAL_STORAGE: Local persistence does not survive container restarts."
}
```

### 2. Readiness Probe Check
```bash
curl -fsS https://<your-render-service>.onrender.com/health/ready
```
**Expected Response:**
```json
{
  "status": "ready",
  "is_ready": true,
  "lifecycle_state": "READY",
  "deployment_mode": "demo",
  "checks": {
    "lifecycle": { "ready": true, "state": "READY" },
    "checkpointer": { "ready": true },
    "telemetry_store": { "ready": true },
    "workspace_lock": { "ready": true },
    "git": { "ready": true },
    "storage_durability": {
      "ready": true,
      "deployment_mode": "demo",
      "persistent": false,
      "notice": "DEMO_EPHEMERAL_STORAGE: State is stored on ephemeral container filesystem and will not survive restarts."
    }
  }
}
```

### 3. Authenticated Tenant Context Inspection
```bash
curl -fsS -H "Authorization: Bearer default-sandbox-key" \
  https://<your-render-service>.onrender.com/api/v1/tenant/context
```
**Expected Response:**
```json
{
  "organization_id": "default-org",
  "organization_name": "Default Organization",
  "user_id": "default-user",
  "user_name": "Default Admin",
  "role": "OWNER",
  "permissions": [ ... ]
}
```

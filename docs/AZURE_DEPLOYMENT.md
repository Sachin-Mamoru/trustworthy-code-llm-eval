# Azure Deployment Documentation

## Deployment Architecture

This document outlines the complete deployment process for the TrustworthyCodeLLM evaluation dashboard on Azure Container Apps.

## Infrastructure Components

### Azure Resources

**Resource Group**: `trustworthy-code-llm-rg`
- Location: East US
- Contains all project resources
- Provides resource management and billing scope

**Container Registry**: `trustworthycodellmacr.azurecr.io`
- SKU: Basic
- Stores Docker images
- Enables automated deployments

**Container Apps Environment**: `trustworthy-code-llm-env`
- Provides compute environment
- Manages networking and scaling
- Handles load balancing

**Container App**: `trustworthy-code-llm-app`
- Runs the FastAPI application
- External ingress enabled
- Auto-scaling configured

## Deployment Process

### Step 1: Resource Creation

```bash
# Create resource group
az group create \
  --name trustworthy-code-llm-rg \
  --location eastus

# Create container registry
az acr create \
  --resource-group trustworthy-code-llm-rg \
  --name trustworthycodellmacr \
  --sku Basic

# Enable admin access
az acr update \
  --name trustworthycodellmacr \
  --admin-enabled true

# Create container apps environment
az containerapp env create \
  --name trustworthy-code-llm-env \
  --resource-group trustworthy-code-llm-rg \
  --location eastus
```

### Step 2: Docker Image Build and Push

```bash
# Login to container registry
az acr login --name trustworthycodellmacr

# Build and push image
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile-llm \
  -t trustworthycodellmacr.azurecr.io/trustworthy-code-llm-dashboard:latest \
  --push .
```

### Step 3: Container App Deployment

```bash
# Deploy container app
az containerapp create \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --environment trustworthy-code-llm-env \
  --image trustworthycodellmacr.azurecr.io/trustworthy-code-llm-dashboard:latest \
  --target-port 3003 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2Gi
```

### Step 4: Environment Configuration

```bash
# Set OpenAI API key
az containerapp update \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --set-env-vars OPENAI_API_KEY="your-api-key-here"
```

## Configuration Details

### Docker Configuration

**Dockerfile-llm**:
- Base image: python:3.9-slim
- Working directory: /app
- Dependencies: requirements-llm.txt
- Exposed port: 3003
- Entry point: uvicorn server

### Application Configuration

**Environment Variables**:
- `OPENAI_API_KEY`: Required for LLM analysis
- `PORT`: Application port (default: 3003)

**Resource Limits**:
- CPU: 1.0 cores
- Memory: 2GB
- Min replicas: 1
- Max replicas: 3

### Network Configuration

**Ingress Settings**:
- External ingress enabled
- Target port: 3003
- HTTPS enabled by default
- Custom domain support available

## Monitoring and Management

### Log Access

```bash
# View recent logs
az containerapp logs show \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --tail 50

# Follow logs in real-time
az containerapp logs show \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --follow
```

### Application Updates

```bash
# Update container image
az containerapp update \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --image trustworthycodellmacr.azurecr.io/trustworthy-code-llm-dashboard:new-tag
```

### Scaling Configuration

```bash
# Update scaling rules
az containerapp update \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --min-replicas 2 \
  --max-replicas 5
```

## Security Configuration

### Access Control

- Container registry requires authentication
- Environment variables securely managed
- HTTPS encryption enforced

### API Key Management

```bash
# Update API key securely
az containerapp update \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --replace-env-vars OPENAI_API_KEY="new-api-key"
```

## Troubleshooting

### Common Issues

1. **Image Build Failures**:
   - Verify Docker is running
   - Check platform architecture (use linux/amd64)
   - Ensure all dependencies are in requirements.txt

2. **Deployment Failures**:
   - Verify Azure CLI authentication
   - Check resource quotas
   - Validate image registry access

3. **Runtime Issues**:
   - Check environment variables
   - Review application logs
   - Verify OpenAI API key validity

### Diagnostic Commands

```bash
# Check container app status
az containerapp show \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg

# List container app revisions
az containerapp revision list \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg

# Get container app URL
az containerapp show \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  --query properties.configuration.ingress.fqdn
```

## Performance Optimization

### Resource Tuning

- Monitor CPU and memory usage
- Adjust scaling parameters based on load
- Optimize container startup time

### Cost Management

- Use appropriate SKU sizes
- Configure auto-scaling rules
- Monitor resource consumption

## Backup and Recovery

### Configuration Backup

```bash
# Export container app configuration
az containerapp show \
  --name trustworthy-code-llm-app \
  --resource-group trustworthy-code-llm-rg \
  > containerapp-config.json
```

### Disaster Recovery

- Container images stored in registry
- Environment variables documented
- Deployment scripts version controlled
- Infrastructure as Code practices

## Development Workflow

### Continuous Deployment

1. Code changes pushed to repository
2. Docker image built and tagged
3. Image pushed to container registry
4. Container app updated with new image
5. Health checks validate deployment

### Version Management

```bash
# Tag and push specific versions
docker tag local-image:latest \
  trustworthycodellmacr.azurecr.io/trustworthy-code-llm-dashboard:v1.0.0

docker push trustworthycodellmacr.azurecr.io/trustworthy-code-llm-dashboard:v1.0.0
```

This deployment architecture provides a robust, scalable, and maintainable foundation for the code evaluation dashboard.

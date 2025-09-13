#!/bin/bash

# Azure Deployment Script for TrustworthyCodeLLM
# This script deploys the production application to Azure Container Apps

set -e

echo "🚀 Starting Azure deployment for TrustworthyCodeLLM..."

# Configuration
RESOURCE_GROUP="trustworthy-code-llm-rg"
LOCATION="eastus"
CONTAINER_APP_ENV="trustworthy-code-env"
CONTAINER_APP_NAME="trustworthy-code-app"
CONTAINER_REGISTRY="trustworthycollm"
IMAGE_NAME="trustworthy-code-llm"
TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo_error "Azure CLI is not installed. Please install it first."
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure (if not already logged in)
echo_info "Checking Azure authentication..."
if ! az account show &> /dev/null; then
    echo_info "Please login to Azure:"
    az login
fi

echo_success "Azure authentication verified"

# Create resource group
echo_info "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create Azure Container Registry
echo_info "Creating Azure Container Registry: $CONTAINER_REGISTRY"
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_REGISTRY \
    --sku Basic \
    --admin-enabled true \
    --output table

# Get registry login server
REGISTRY_SERVER=$(az acr show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
echo_success "Registry server: $REGISTRY_SERVER"

# Build and push Docker image
echo_info "Building Docker image..."
az acr build \
    --registry $CONTAINER_REGISTRY \
    --image $IMAGE_NAME:$TAG \
    --file Dockerfile \
    .

echo_success "Docker image built and pushed"

# Create Container Apps environment
echo_info "Creating Container Apps environment: $CONTAINER_APP_ENV"
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Get ACR credentials
REGISTRY_USERNAME=$(az acr credential show --name $CONTAINER_REGISTRY --query username --output tsv)
REGISTRY_PASSWORD=$(az acr credential show --name $CONTAINER_REGISTRY --query passwords[0].value --output tsv)

# Create the container app
echo_info "Deploying container app: $CONTAINER_APP_NAME"
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image "$REGISTRY_SERVER/$IMAGE_NAME:$TAG" \
    --registry-server $REGISTRY_SERVER \
    --registry-username $REGISTRY_USERNAME \
    --registry-password $REGISTRY_PASSWORD \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars \
        "AZURE_OPENAI_KEY=secretref:azure-openai-key" \
        "AZURE_OPENAI_ENDPOINT=secretref:azure-openai-endpoint" \
        "AZURE_OPENAI_MODEL=gpt-4" \
        "OPENAI_API_KEY=secretref:openai-api-key" \
        "ANTHROPIC_API_KEY=secretref:anthropic-api-key" \
        "ENVIRONMENT=production" \
    --output table

# Get the application URL
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo_success "Deployment completed!"
echo ""
echo "🌐 Application URL: https://$APP_URL"
echo ""
echo "📋 Next steps:"
echo "1. Set up secrets for API keys:"
echo "   az containerapp secret set --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP \\"
echo "     --secrets azure-openai-key=YOUR_AZURE_OPENAI_KEY \\"
echo "               azure-openai-endpoint=YOUR_AZURE_OPENAI_ENDPOINT \\"
echo "               openai-api-key=YOUR_OPENAI_API_KEY \\"
echo "               anthropic-api-key=YOUR_ANTHROPIC_API_KEY"
echo ""
echo "2. Test the deployment:"
echo "   curl https://$APP_URL/health"
echo ""
echo "3. Monitor the application:"
echo "   az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "📊 Dashboard: https://$APP_URL"
echo "📖 API Docs: https://$APP_URL/api/docs"
echo ""
echo_success "TrustworthyCodeLLM is now live on Azure! 🎉"

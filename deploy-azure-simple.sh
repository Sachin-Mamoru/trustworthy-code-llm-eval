#!/bin/bash

# Azure deployment script for TrustworthyCodeLLM Production
set -e

echo "🚀 Deploying TrustworthyCodeLLM to Azure..."

# Configuration
RESOURCE_GROUP="trustworthy-code-llm-rg"
LOCATION="eastus"
ACR_NAME="trustworthycodellmacr"
APP_NAME="trustworthy-code-llm"
CONTAINER_APP_ENV="trustworthy-code-env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if user is logged into Azure
print_status "Checking Azure login status..."
if ! az account show &> /dev/null; then
    print_error "Please login to Azure first: az login"
    exit 1
fi

print_success "Azure login verified"

# Create resource group
print_status "Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Create Azure Container Registry
print_status "Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output table

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
print_success "ACR created: $ACR_LOGIN_SERVER"

# Build and push container image
print_status "Building and pushing container image..."
az acr build \
    --registry $ACR_NAME \
    --image "trustworthy-code-llm:latest" \
    --file Dockerfile-simple \
    .

# Create Container Apps environment
print_status "Creating Container Apps environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Create Container App
print_status "Creating Container App..."
az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image "$ACR_LOGIN_SERVER/trustworthy-code-llm:latest" \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 80 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --output table

# Get the application URL
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv)

print_success "🎉 Deployment completed successfully!"
echo ""
echo "==============================================="
echo "📋 Deployment Summary"
echo "==============================================="
echo "🌐 Application URL: https://$APP_URL"
echo "🏥 Health Check: https://$APP_URL/health"
echo "📊 Resource Group: $RESOURCE_GROUP"
echo "🐳 Container Registry: $ACR_LOGIN_SERVER"
echo "💻 Container App: $APP_NAME"
echo "🌍 Location: $LOCATION"
echo "==============================================="
echo ""
print_success "Your supervisor can now access the application at: https://$APP_URL"
echo ""
print_warning "Note: It may take a few minutes for the app to be fully accessible."
echo ""
echo "🔧 To update the application:"
echo "   1. Make code changes"
echo "   2. Run: az acr build --registry $ACR_NAME --image trustworthy-code-llm:latest --file Dockerfile-simple ."
echo "   3. Run: az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --image $ACR_LOGIN_SERVER/trustworthy-code-llm:latest"
echo ""
echo "🗑️  To clean up resources:"
echo "   az group delete --name $RESOURCE_GROUP --yes --no-wait"

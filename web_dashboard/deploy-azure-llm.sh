#!/bin/bash

# LLM-Enhanced Azure Deployment Script
# This deploys the dashboard with OpenAI integration

set -e

echo "🚀 Deploying TrustworthyCodeLLM with LLM Support to Azure..."

# Configuration
RESOURCE_GROUP="trustworthy-code-llm-rg"
LOCATION="eastus"
REGISTRY_NAME="trustworthycodellmacr"
APP_NAME="trustworthy-code-llm-app"
CONTAINER_APP_ENV="trustworthy-code-llm-env"
IMAGE_NAME="trustworthy-code-llm-dashboard"
TAG="llm-$(date +%Y%m%d-%H%M%S)"

echo "📋 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Registry: $REGISTRY_NAME"
echo "  App Name: $APP_NAME"
echo "  Image: $IMAGE_NAME:$TAG"

# Check if logged in to Azure
echo "🔐 Checking Azure login..."
if ! az account show > /dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run 'az login' first."
    exit 1
fi

echo "✅ Azure login confirmed"

# Create resource group if it doesn't exist
echo "📁 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create Azure Container Registry if it doesn't exist
echo "🐳 Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true --output table

# Get ACR login server
ACR_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
echo "📍 ACR Server: $ACR_SERVER"

# Build and push Docker image
echo "🔨 Building Docker image..."
docker build -f Dockerfile-llm -t $ACR_SERVER/$IMAGE_NAME:$TAG .

echo "📤 Pushing to Azure Container Registry..."
az acr login --name $REGISTRY_NAME
docker push $ACR_SERVER/$IMAGE_NAME:$TAG

# Create Container Apps environment if it doesn't exist
echo "🌐 Creating Container Apps environment..."
if ! az containerapp env show --name $CONTAINER_APP_ENV --resource-group $RESOURCE_GROUP > /dev/null 2>&1; then
    az containerapp env create --name $CONTAINER_APP_ENV --resource-group $RESOURCE_GROUP --location $LOCATION --output table
fi

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Deploy Container App with environment variables
echo "🚀 Deploying Container App with LLM support..."

# Prompt for OpenAI API key
echo ""
echo "🔑 OpenAI API Key Configuration:"
echo "To enable LLM analysis, you need an OpenAI API key."
echo "Get one from: https://platform.openai.com/api-keys"
echo ""
read -p "Enter your OpenAI API key (or press Enter to skip): " OPENAI_API_KEY

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  No API key provided - deploying with AST analysis only"
    az containerapp create \\
        --name $APP_NAME \\
        --resource-group $RESOURCE_GROUP \\
        --environment $CONTAINER_APP_ENV \\
        --image $ACR_SERVER/$IMAGE_NAME:$TAG \\
        --registry-server $ACR_SERVER \\
        --registry-username $ACR_USERNAME \\
        --registry-password $ACR_PASSWORD \\
        --target-port 8080 \\
        --ingress 'external' \\
        --cpu 0.5 \\
        --memory 1Gi \\
        --min-replicas 1 \\
        --max-replicas 3 \\
        --output table
else
    echo "✅ Deploying with OpenAI API key configured"
    az containerapp create \\
        --name $APP_NAME \\
        --resource-group $RESOURCE_GROUP \\
        --environment $CONTAINER_APP_ENV \\
        --image $ACR_SERVER/$IMAGE_NAME:$TAG \\
        --registry-server $ACR_SERVER \\
        --registry-username $ACR_USERNAME \\
        --registry-password $ACR_PASSWORD \\
        --target-port 8080 \\
        --ingress 'external' \\
        --cpu 0.5 \\
        --memory 1Gi \\
        --min-replicas 1 \\
        --max-replicas 3 \\
        --env-vars "OPENAI_API_KEY=$OPENAI_API_KEY" \\
        --output table
fi

# Get the app URL
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv)

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📊 Dashboard URL: https://$APP_URL"
echo ""
echo "🔧 Features:"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "  • AST-based code analysis"
    echo "  • Structure and complexity metrics"
    echo "  • ⚠️  LLM analysis disabled (no API key)"
else
    echo "  • AST-based code analysis"
    echo "  • OpenAI LLM analysis (GPT-3.5-turbo)"
    echo "  • Combined scoring system"
fi
echo ""
echo "🔑 To add/update OpenAI API key later:"
echo "az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --set-env-vars OPENAI_API_KEY=your_key_here"
echo ""
echo "📝 To view logs:"
echo "az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "🗑️  To delete everything:"
echo "az group delete --name $RESOURCE_GROUP --yes --no-wait"

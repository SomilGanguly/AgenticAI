FROM mcr.microsoft.com/azure-functions/python:4-python3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Terraform
ENV TERRAFORM_VERSION=1.12.1
RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor > /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list \
    && apt-get update && apt-get install -y terraform=${TERRAFORM_VERSION}-* \
    && terraform --version

# Set Python environment
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    FUNCTIONS_WORKER_RUNTIME=python \
    TF_IN_AUTOMATION=true \
    TF_INPUT=false

# Copy function app
COPY . /home/site/wwwroot

# Install Python dependencies
RUN cd /home/site/wwwroot && \
    pip install --no-cache-dir -r requirements.txt
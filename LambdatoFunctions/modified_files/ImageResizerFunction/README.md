# Azure Function: Image Resizer

## Overview

This Azure Function automatically resizes JPEG and PNG images uploaded to an Azure Blob Storage container. The resized image is saved to an output container with a `resized-` prefix, replicating the functionality of an AWS Lambda triggered by S3.

## Folder Structure

```
ImageResizerFunction/
├── src/main/java/com/example/ImageResizerFunction.java
├── function.json
├── host.json
├── local.settings.json
├── pom.xml
├── .gitignore
└── README.md
```

## Prerequisites
- Java 11 or higher
- Azure CLI
- Maven
- An Azure subscription
- An Azure Storage account with `input-container` and `output-container` containers

## Deployment Steps
1. **Clone the Repository**

2. **Configure local.settings.json**
   - Set your `AzureWebJobsStorage` connection string
   - Confirm `FUNCTIONS_WORKER_RUNTIME` is `java`
   - Set `OUTPUT_CONTAINER` appropriately (default: `output-container`)

3. **Build the Function App**
   ```sh
   mvn clean package
   ```

4. **Log in to Azure and Create Resources**
   ```sh
   az login
   az group create --name ImageResizerRG --location <your-region>
   az storage account create --name <yourstorageaccount> --resource-group ImageResizerRG --sku Standard_LRS
   az storage container create --account-name <yourstorageaccount> --name input-container
   az storage container create --account-name <yourstorageaccount> --name output-container
   az functionapp create --resource-group ImageResizerRG --consumption-plan-location <your-region> --runtime java --runtime-version 11 --functions-version 4 --name image-resizer-func --storage-account <yourstorageaccount>
   ```

5. **Deploy the Function**
   ```sh
   mvn azure-functions:deploy
   ```

6. **Assign Managed Identity (Optional for Key Vault/secure scenarios)**
   ```sh
   az functionapp identity assign --name image-resizer-func --resource-group ImageResizerRG
   az role assignment create --assignee <principalId> --role "Storage Blob Data Contributor" --scope <storage account resource id>
   ```

7. **Upload a Test Image**
   - Upload a .jpg or .png image to the `input-container`. The resized version appears in the `output-container` as `resized-<original-name>`.

## Monitoring & Logging
- Application Insights is automatically enabled if configured in the Azure portal.

## Configuration
- Blob containers and output details can be modified in `local.settings.json` or Azure App Settings post-deployment.
- Max resize dimension is hardcoded as 100px. To change, modify `MAX_DIM` in code and redeploy.

## Security
- Use Managed Identity for secure storage access.
- Do not store sensitive info in code or version control.

## Cleanup
```sh
az group delete --name ImageResizerRG --yes --no-wait
```

---

_Migration from AWS Lambda S3 image resizer implemented for Azure Functions (Java)._

# Azure Function: Image Resizer (Blob Trigger)

This Azure Function resizes images uploaded to Azure Blob Storage. It is a migration of the original AWS Lambda application triggered by S3 events. Images are processed in memory and the resized versions are saved back to Blob Storage with a `resized-` prefix.

## Project Structure

- src/main/java/example/Handler.java      - Main Azure Function code
- function.json                          - Function runtime/binding configuration
- host.json                              - Global host configuration
- local.settings.json                    - Local development settings (do not commit secrets)
- pom.xml                                - Java/Maven build and dependency file
- .gitignore                             - Ignore build outputs and local configs
- README.md                              - This file

## Prerequisites
- Java 21 SDK
- Maven 3.6+
- Azure CLI
- Azure subscription
- Function Core Tools (for local run)

## Deploying to Azure

1. **Create Azure resources**
   - Storage Account, Function App (Java 21), Application Insights (see ARM/Bicep example below)

2. **Assign Managed Identity**
   - Assign "Storage Blob Data Contributor" to your Function App's System Assigned Managed Identity for the target Storage Account.

3. **Build and Deploy**
```shell
mvn clean package
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

4. **Configure Application Settings**
   - AzureWebJobsStorage: Set to your Storage connection string
   - APPLICATIONINSIGHTS_CONNECTION_STRING: Set to your AI instance

5. **Test**
   - Upload `.jpg` or `.png` to the `images` container in your Storage Account
   - A new blob prefixed with `resized-` will appear

## Local Development
1. Set your connection string in `local.settings.json` (see template file)
2. Run with
```shell
func start
```

## ARM/Bicep Sample (Provision storage, function app, insights)
_See template in migration plan above._

## Post-Migration Steps
- Validate Managed Identity assignment in Azure Portal
- Confirm blob trigger path and prefix match expectations
- Monitor logs in Application Insights

## Performance & Security
- Function set for Java 21, 512MB default (can be adjusted in portal)
- All secrets/keys reside in Azure settings or Key Vault (not in code)

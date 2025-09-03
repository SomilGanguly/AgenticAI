terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">=3.0.0"
    }
  }
}

# provider "azuread" {
#   tenant_id = var.tenant_id
#   use_cli = true
# }

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
  storage_use_azuread = true
  subscription_id = var.wl_subscription_id
}

# Configure the Microsoft Azure Provider pointing to the Hub subscription
provider "azurerm" {
  features {}
  alias           = "hubsubscription"
  subscription_id = var.hub.subscription_id
}

# New provider for Oracle subscription (selects subscription based on environment)
provider "azurerm" {
  features {}
  alias           = "odaa"
  #subscription_id = (lower(var.environment) == "prod" || lower(var.environment) == "prd") ? var.odaa_peering["prd"].subscription_id : var.odaa_peering["nprd"].subscription_id
  #subscription_id = var.odaa_peering[local.odaa_env_key].subscription_id
  subscription_id = var.is_odaa_peering_enabled ? var.odaa_peering[local.odaa_env_key].subscription_id : var.wl_subscription_id
}

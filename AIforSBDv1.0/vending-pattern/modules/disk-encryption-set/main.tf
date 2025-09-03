resource "azurerm_disk_encryption_set" "this" {
  name                      = var.disk_encryption_set_name
  resource_group_name       = var.resource_group_name
  location                  = var.location
  auto_key_rotation_enabled = var.auto_key_rotation_enabled
 
  identity {
    type         = "UserAssigned"
    identity_ids = var.uai_ids
  }
  key_vault_key_id = var.key_vault_key_id
}
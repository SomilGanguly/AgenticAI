data "azurerm_client_config" "this" {}

data "azurerm_virtual_network" "hubvnet" {
  provider            = azurerm.hubsubscription
  name                = var.hub.vnet_name
  resource_group_name = var.hub.hub_rg_name
}

data "azurerm_subnet" "hubsubnets" {
  provider             = azurerm.hubsubscription
  for_each             = local.hubsubnets_map
  name                 = each.value 
  virtual_network_name = var.hub.vnet_name
  resource_group_name  = var.hub.hub_rg_name
}

data "azurerm_firewall" "hubfw" {
  provider            = azurerm.hubsubscription
  name                = var.hub.firewall_name
  resource_group_name = var.hub.hub_rg_name
}

# data "azuread_group" "group_objects" {
#   provider         = azuread
#   for_each         = var.rbac_info_groupname
#   display_name     = each.key
#   security_enabled = true
# }

# data "azuread_service_principal" "mi_objects" {
#   provider         = azuread
#   for_each         = var.rbac_info_mi
#   display_name     = each.key
# }

data "azurerm_subscription" "current" {
  subscription_id = var.wl_subscription_id
}

# New data block for Oracle VNET (only if is_odaa_peering_enabled is true)
data "azurerm_virtual_network" "odaa" {
 count               = var.is_odaa_peering_enabled? 1 : 0
 provider            = azurerm.odaa
 name                = var.odaa_peering[local.odaa_env_key].odaa_vnet_name
 resource_group_name = var.odaa_peering[local.odaa_env_key].odaa_rg_name
}



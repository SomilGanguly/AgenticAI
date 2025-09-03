module "lz_vending" {
  source  = "Azure/lz-vending/azurerm"
  version = "4.1.5" # change this to your desired version, https://www.terraform.io/language/expressions/version-constraints

  location = var.location

  # subscription variables
  subscription_id = var.wl_subscription_id
  
  role_assignment_enabled = false
  # role_assignments        = local.role_assignments_all
  #role_assignment_enabled = length(var.rbac_info_groupname) == 0 && length(var.rbac_info_principalid) == 0 ? false : true
  #role_assignments        = length(var.rbac_info_groupname) == 0 ? local.role_assignments_principalid : local.role_assignments_groupname

  network_watcher_resource_group_enabled = false
  # resource group variables
  resource_group_creation_enabled = true
  resource_groups                 = local.resource_groups


  # virtual network variables
  virtual_network_enabled = true
  virtual_networks        = local.virtual_networks

}


module "avm-res-network-networksecuritygroup" {
  depends_on          = [module.lz_vending]
  source              = "Azure/avm-res-network-networksecuritygroup/azurerm"
  version             = "0.3.0"
  for_each            = { for k, v in local.network_security_groups : k => v if local.subnets[k].has_nsg } #NSG is only created if has_nsg in true
  enable_telemetry    = local.enable_telemetry
  resource_group_name = local.virtual_networks["vnet1"].resource_group_name
  name                = each.value.name
  security_rules      = each.value.security_rules
  location            = var.location
}

module "avm-res-network-routetable" {
  depends_on          = [module.lz_vending]
  source              = "Azure/avm-res-network-routetable/azurerm"
  version             = "0.2.2"
  for_each            = local.route_tables
  enable_telemetry    = local.enable_telemetry
  name                = each.value.name
  resource_group_name = local.virtual_networks["vnet1"].resource_group_name
  location            = var.location
  routes              = each.value.routes
}


module "avm-res-network-subnet" {
  source   = "Azure/avm-res-network-virtualnetwork/azurerm//modules/subnet"
  version = "0.8.1"
  for_each = local.subnets
  virtual_network = {
    resource_id = module.lz_vending.virtual_network_resource_ids["vnet1"]
  }
  name           = each.value.name
  address_prefix = each.value.address_prefix

  # Attach NSG only if has_nsg is true
  network_security_group = each.value.has_nsg ? {
    id = module.avm-res-network-networksecuritygroup[each.key].resource_id
  } : null

  private_endpoint_network_policies = "Enabled"
  private_link_service_network_policies_enabled = "true"
  route_table = {
    id = module.avm-res-network-routetable["tohub"].resource_id
  }
  default_outbound_access_enabled = true
  # Apply delegation only if defined
  delegation = each.value.delegation
  depends_on = [module.lz_vending, module.avm-res-network-networksecuritygroup, module.avm-res-network-routetable]
}

# Oracle VNET Peering Resources (only created if is_odaa_peering_enabled is true)
resource "azurerm_virtual_network_peering" "local_to_odaa" {
 count                     = var.is_odaa_peering_enabled ? 1 : 0
 name                      = "peer_${local.vnet_name}_${data.azurerm_virtual_network.odaa[0].name}"
 resource_group_name       = local.virtual_networks["vnet1"].resource_group_name
 virtual_network_name      = local.virtual_networks["vnet1"].name
 remote_virtual_network_id = data.azurerm_virtual_network.odaa[0].id
 
 allow_forwarded_traffic      = true
 allow_virtual_network_access = true
 use_remote_gateways          = false
 allow_gateway_transit        = false
}
 
resource "azurerm_virtual_network_peering" "odaa_to_local" {
 count                     = var.is_odaa_peering_enabled? 1 : 0
 provider                  = azurerm.odaa
 name                      = "peer_${data.azurerm_virtual_network.odaa[0].name}_${local.vnet_name}"
 resource_group_name       = data.azurerm_virtual_network.odaa[0].resource_group_name  ##var.odaa_peering[local.odaa_env_key].odaa_rg_name
 virtual_network_name      = data.azurerm_virtual_network.odaa[0].name                 ##var.odaa_peering[local.odaa_env_key].odaa_vnet_name
 remote_virtual_network_id = module.lz_vending.virtual_network_resource_ids["vnet1"]
 
 allow_forwarded_traffic      = true
 allow_virtual_network_access = true
 use_remote_gateways          = false
 allow_gateway_transit        = false
}

module "avm-res-storage-storageaccount" {
  depends_on                    = [module.lz_vending]
  for_each                      = local.storage_accounts
  source                        = "Azure/avm-res-storage-storageaccount/azurerm"
  version                       = "0.5.0"
  enable_telemetry              = local.enable_telemetry
  account_replication_type      = "LRS"
  account_tier                  = "Standard"
  account_kind                  = "StorageV2"
  location                      = var.location
  name                          = each.value.name
  https_traffic_only_enabled    = true
  resource_group_name           = local.resource_groups["sarg"].name
  min_tls_version               = "TLS1_2"
  shared_access_key_enabled     = false
  public_network_access_enabled = true
  # managed_identities = {
  #   system_assigned            = false
  #   user_assigned_resource_ids = [local.storage_cmk[each.key].user_assigned_identity_id]
  # }
  # customer_managed_key = {
  #   key_vault_resource_id     = local.storage_cmk[each.key].key_vault_resource_id
  #   key_name                  = local.storage_cmk[each.key].key_name
  #   user_assigned_identity = {
  #     resource_id = local.storage_cmk[each.key].user_assigned_identity_id
  #   }
  # }
  blob_properties = {
    versioning_enabled = true
  }

  /* role_assignments = {
    role_assignment_1 = {
      role_definition_id_or_name       = data.azurerm_role_definition.example.name
      principal_id                     = coalesce(var.msi_id, data.azurerm_client_config.current.object_id)
      skip_service_principal_aad_check = false
    },
    role_assignment_2 = {
      role_definition_id_or_name       = "Owner"
      principal_id                     = data.azurerm_client_config.current.object_id
      skip_service_principal_aad_check = false
    },

  } */
 network_rules = {
    bypass                     = ["AzureServices"]
    default_action             = "Deny"
    ip_rules                   = var.storageaccount_ip_rules
    virtual_network_subnet_ids = []
  }

}

module "keyvault" {
  depends_on               = [module.lz_vending]
  for_each                 = local.key_vaults
  source                   = "Azure/avm-res-keyvault-vault/azurerm"
  version                  = "0.9.1"
  name                     = each.value.name
  enable_telemetry         = local.enable_telemetry
  location                 = var.location
  sku_name                 = "standard"
  resource_group_name      = local.resource_groups["akvrg"].name
  tenant_id                = data.azurerm_client_config.this.tenant_id
  public_network_access_enabled = true
    network_acls             = {
    bypass = "AzureServices"
    default_action = "Deny"
    ip_rules = local.akv_network_rules_ip_list
  }
  purge_protection_enabled = true
}
 

# module "disk_encryption_set" {
#   depends_on               = [module.lz_vending]
#   source = "./modules/disk-encryption-set"
#   for_each                    = local.disk_encryption_sets
#   disk_encryption_set_name    = each.value.name
#   resource_group_name         = local.resource_groups["desrg"].name
#   location                    = var.location
#   key_vault_key_id            = each.value.key_vault_key_id
#   uai_ids                     = [each.value.uai_id]
#   auto_key_rotation_enabled   = each.value.auto_key_rotation_enabled
# }


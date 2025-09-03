
locals {
  odaa_env_key = (lower(var.environment) == "prod" || lower(var.environment) == "prd") ? "prd" : "nprd"
  location_short   = local.location_map[var.location]
  enable_telemetry = false
  hubsubnets = ["AzureFirewallManagementSubnet", "AzureFirewallSubnet"]
  hubsubnets_merged = concat(local.hubsubnets, var.overwrite_route_subnets_list)
  hubsubnets_map = zipmap(range(length(local.hubsubnets_merged)), local.hubsubnets_merged)
  inherited_tags_list = [ "business_service", "confidentiality", "environment", "managed_by", "service_class", "support_group" ]
  inherited_tags = {
    for k, v in data.azurerm_subscription.current.tags : k => v if contains(local.inherited_tags_list, k)
  }

  # role_assignments_groupname = merge([
  #   for group, roles in var.rbac_info_groupname : {
  #     for role in roles :
  #     "${group}-${role}" => {
  #       "principal_id" = data.azuread_group.group_objects[group].object_id
  #       "definition"   = role
  #     }
  #   }
  # ]...)

  # role_assignments_mi = merge([
  #   for mi, roles in var.rbac_info_mi : {
  #     for role in roles :
  #     "${mi}-${role}" => {
  #       "principal_id" = data.azuread_service_principal.mi_objects[mi].object_id
  #       "definition"   = role
  #     }
  #   }
  # ]...)

  # role_assignments_principalid = merge([
  #   for group, roles in var.rbac_info_principalid : {
  #     for role in roles :
  #     "${group}-${role}" => {
  #       "principal_id" = group
  #       "definition"   = role
  #     }
  #   }
  # ]...)

  role_assignments_all = []

  resource_groups_default_resources = {
    sarg = {
      name     = "rg-azu${local.location_short}-${var.appId}-sa-${var.environment}-001"
      location = var.location
      tags     = local.inherited_tags
    }
    akvrg = {
      name     = "rg-azu${local.location_short}-${var.appId}-kv-${var.environment}-001"
      location = var.location
      tags     = local.inherited_tags
    }
    desrg = {
      name     = "rg-azu${local.location_short}-${var.appId}-des-${var.environment}-001"
      location = var.location
      tags     = local.inherited_tags
    }
  }

  resource_groups_tiers = {
    for subs in var.subnet : subs.tier =>
    {
      name     = "rg-azu${local.location_short}-${var.appId}-${subs.tier}-${var.environment}-001"
      location = var.location
      tags     = merge(local.inherited_tags, { az_backup = var.az_backup_enable})
    }
  }

  resource_groups = merge(local.resource_groups_default_resources, local.resource_groups_tiers)

  vnet_name = "vnet-azu${local.location_short}-${var.rd_id}-${var.appId}-${var.environment}-001"

  virtual_networks = {
    vnet1 = {
      name                        = local.vnet_name
      address_space               = var.vnet_ip_range
      resource_group_name         = "rg-azu${local.location_short}-${var.appId}-netw-${var.environment}-001"
      resource_group_lock_enabled = false
      hub_peering_enabled         = true
      hub_network_resource_id     = data.azurerm_virtual_network.hubvnet.id
      hub_peering_name_tohub      = "peer_${local.vnet_name}_${var.hub.vnet_name}"
      hub_peering_name_fromhub    = "peer_${var.hub.vnet_name}_${local.vnet_name}"
      dns_servers                 = [data.azurerm_firewall.hubfw.ip_configuration[0].private_ip_address]
      ddos_protection_enabled     = false
      # ddos_protection_plan_id     = "/subscriptions/b16eb1c0-7284-457b-92cf-a9713bac81d1/resourceGroups/rg-azuuks-ddos-prd-001/providers/Microsoft.Network/ddosProtectionPlans/ddos-azuuks-plan-prd-001"
    }
  }

  subnet_count = length(var.subnet) # Get the number of subnets
 
  subnets = {
    for subs in var.subnet : subs.tier =>
    {
      name           = "snet-azu${local.location_short}-${subs.security_zone}-${replace(replace(subs.address, ".", "_"), "/", "-")}"
      address_prefix = "${subs.address}"
 
      # Apply delegation if:
      # - Only 1 subnet exists, apply delegation to that subnet
      # - Multiple subnets exist, apply delegation only if tier is "db"
      delegation = (
        var.is_oracle_subscription &&
        (local.subnet_count == 1 || lower(subs.tier) == "db")
      ) ? [{
        name = "oracle-delegation"
        service_delegation = {
          name = "Oracle.Database/networkAttachments"
        }
      }] : []

      #Exclude NSG attachmeny only if the subnet is delegated for Oracle - https://learn.microsoft.com/en-us/azure/oracle/oracle-db/oracle-database-network-plan#constraints
      has_nsg = !(
        var.is_oracle_subscription &&
        (local.subnet_count == 1 || lower(subs.tier) == "db")
      )
    }
  }

  overwrite_routes = {
    for k, v in data.azurerm_subnet.hubsubnets : "overwrite-route-${index(keys(data.azurerm_subnet.hubsubnets), k) + 1}" => {
      name                   = "udr-00101-00${index(keys(data.azurerm_subnet.hubsubnets), k) + 2}"
      address_prefix         = "${v.address_prefixes[0]}"
      next_hop_type          = "VirtualAppliance"
      next_hop_in_ip_address = data.azurerm_firewall.hubfw.ip_configuration[0].private_ip_address
    }  
  }

  route_tables = {
    tohub = {
      name = "rt-azu${local.location_short}-${var.rd_id}-${var.appId}-wl-${var.environment}-001"
      routes = merge({
        default-route = {
          name                   = "udr-00101-001"
          address_prefix         = "0.0.0.0/0"
          next_hop_type          = "VirtualAppliance"
          next_hop_in_ip_address = data.azurerm_firewall.hubfw.ip_configuration[0].private_ip_address
        }}, local.overwrite_routes)    
    }
  }

  subnets_by_zone = {
    for s in var.subnet : s.security_zone => s...
  }

  network_security_groups = merge([
    for zone, subnets in local.subnets_by_zone : {
      for idx, s in subnets : s.tier => {
        name = "nsg-azu${local.location_short}-${var.rd_id}-${var.appId}-${s.security_zone}-${var.environment}-${format("%03d", idx + 1)}"
        security_zone = s.security_zone
        settings = {
          inc = "${format("%03d", idx + 1)}"
        }
        security_rules = merge(s.network_security_rules,
        {
          DenyIntraVnetAddressSpace = {
            access                     = "Deny"
            direction                  = "Inbound"
            name                       = "DenyIntraVnetAddressSpace"
            priority                   = 4000
            source_address_prefixes    = var.vnet_ip_range
            destination_address_prefix = "*"
            source_port_range          = "*"
            destination_port_range     = "*"
            protocol                   = "*"
          }
          AllowIntraSubnet = {
            access                     = "Allow"
            direction                  = "Inbound"
            name                       = "AllowIntraSubnet"
            priority                   = 3500
            source_address_prefix      = s.address
            destination_address_prefix = "*"
            source_port_range          = "*"
            destination_port_range     = "*"
            protocol                   = "*"
          }
        })
      }
    }
  ]...)

  storage_accounts = {
    sa1 = {
      name               = "st${local.location_short}${var.appId}${var.environment}001"
      azmigcontainername = "azmig-container-${var.appId}"
      inc = "001"
    }
  }

  key_vaults = {
    kv1 = {
      name = "kv${local.location_short}${var.appId}${var.environment}001"
      settings = {
        inc = "001"
      }
    }
  }

  # disk_encryption_sets = {
  #   des1 = {
  #     name = "des-azu${local.location_short}-${var.appId}-${var.environment}-001"
  #     key_vault_key_id = "https://${var.cmk.keyvault_name}.vault.azure.net/keys/${var.cmk.vm_key_name}"
  #     uai_id = "/subscriptions/${var.cmk.subscription_id}/resourceGroups/${var.cmk.rg_name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${var.cmk.uai_name}"
  #     auto_key_rotation_enabled = true
  #   }
  # }

  # storage_cmk = {
  #   for sa_key in keys(local.storage_accounts) : sa_key => {
  #     key_vault_resource_id = "/subscriptions/${var.cmk.subscription_id}/resourceGroups/${var.cmk.rg_name}/providers/Microsoft.KeyVault/vaults/${var.cmk.keyvault_name}"
  #     key_name     = var.cmk.st_key_name
  #     user_assigned_identity_id = "/subscriptions/${var.cmk.subscription_id}/resourceGroups/${var.cmk.rg_name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${var.cmk.uai_name}"
  #   }
  # }

  location_map = {
    "Australia Central 2"  = "au2",
    "Australia Central"    = "auc",
    "Australia East"       = "aue",
    "Australia Southeast"  = "ase",
    "australiacentral2"    = "au2",
    "australiacentral"     = "auc",
    "australiaeast"        = "aue",
    "australiasoutheast"   = "ase",
    "Brazil South"         = "brs",
    "brazilsouth"          = "brs",
    "Canada Central"       = "cac",
    "Canada East"          = "cae",
    "canadacentral"        = "cac",
    "canadaeast"           = "cae",
    "Central India"        = "cin",
    "Central US"           = "cus",
    "centralindia"         = "cin",
    "centralus"            = "cus",
    "East Asia"            = "eas",
    "East US 2"            = "eu2",
    "East US"              = "eus",
    "eastasia"             = "eas",
    "eastus"               = "eus",
    "eastus2"              = "eu2",
    "France Central"       = "frc",
    "France South"         = "frs",
    "francecentral"        = "frc",
    "francesouth"          = "frs",
    "Germany North"        = "gno",
    "Germany West Central" = "gwc",
    "germanynorth"         = "gno",
    "germanywestcentral"   = "gwc",
    "Italy North"          = "itn",
    "Japan East"           = "jae",
    "Japan West"           = "jaw",
    "japaneast"            = "jae",
    "japanwest"            = "jaw",
    "Korea Central"        = "krc",
    "Korea South"          = "kos",
    "koreacentral"         = "krc",
    "koreasouth"           = "kos",
    "North Central US"     = "ncu",
    "North Europe"         = "neu",
    "northcentralus"       = "ncu",
    "northeurope"          = "neu",
    "South Africa North"   = "san",
    "South Africa West"    = "saw",
    "South Central US"     = "scu",
    "South India"          = "sin",
    "southafricanorth"     = "san",
    "southafricawest"      = "saw",
    "southcentralus"       = "scu",
    "Southeast Asia"       = "sea",
    "southeastasia"        = "sea",
    "southindia"           = "sin",
    "UAE Central"          = "uac",
    "UAE North"            = "uan",
    "uaecentral"           = "uac",
    "uaenorth"             = "uan",
    "UK South"             = "uks",
    "UK West"              = "ukw",
    "uksouth"              = "uks",
    "ukwest"               = "ukw",
    "West Central US"      = "wcus",
    "West Europe"          = "weu",
    "West India"           = "win",
    "West US 2"            = "wu2",
    "West US"              = "wus",
    "westcentralus"        = "wcus",
    "westeurope"           = "weu",
    "westindia"            = "win",
    "westus"               = "wus",
    "westus2"              = "wu2"
  }

}

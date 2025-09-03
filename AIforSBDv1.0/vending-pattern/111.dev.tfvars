appId         = "111"
rd_id         = "130"
environment   = "dev"
location      = "uksouth"
vnet_ip_range = ["10.147.133.192/26"]

subnet = [
  {
    tier          = "agw"
    address       = "10.147.133.192/27"
    security_zone = "tp"
    network_security_rules = {
    }
  },
  {
    tier          = "presentation"
    address       = "10.147.133.224/29"
    security_zone = "tp"
    network_security_rules = {
    }
  },
  {
    tier          = "app"
    address       = "10.147.133.232/29"
    security_zone = "ta"
    network_security_rules = {
    }
  },
  {
    tier          = "db"
    address       = "10.147.133.240/29"
    security_zone = "td"
    network_security_rules = {
    }
  }
]

# overwrite_route_subnets_list = ["snet-azuuks-c1-10_147_131_16-28"]

rbac_info_groupname = {
  "ADM-VF-Azure-UK-MSMigration-Admins" = ["Reader", "Virtual Machine Contributor", "Storage Blob Data Contributor", "Key Vault Secrets User"]
}
rbac_info_mi = {
  "id-azuuks-mcs-azmigrate" = ["Contributor", "Storage Blob Data Contributor", "Key Vault Secrets Officer"]
  "id-azuuks-mcs-terraform" = ["Storage Blob Data Contributor", "Key Vault Secrets Officer"]
}

storageaccount_ip_rules = ["185.69.146.0/24"]

hub = {
  subscription_id = "177c1f84-4b2a-414d-974e-5fbdf5d84250"
  hub_rg_name     = "myResourceGroup"
  vnet_name       = "vneto"
  firewall_name   = "firewallo"
}
# cmk = {
#   subscription_id = "177c1f84-4b2a-414d-974e-5fbdf5d84250"
#   rg_name = "myResourceGroup"
#   keyvault_name = "kvuksukcmkprd01"
#   vm_key_name = "kvk-azuuks-cmkvm-prd-001"
#   st_key_name = "kvk-azuuks-cmkst-prd-01"
#   uai_name = "id-azuuks-cmkrsv-prd-001"
# }

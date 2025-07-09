variable "appId" {
  type        = string
  description = "Application ID of the deployment"
}

variable "rd_id" {
  type        = string
  description = <<DESCRIPTION
DESCRIPTION
}

variable "environment" {
  type        = string
  description = "The environment (e.g., development, staging, production) for the deployment."
}

variable "location" {
  type        = string
  description = "The Azure location where the resources will be deployed."
  nullable    = false
}

variable "vnet_ip_range" {
  type        = list(string)
  description = <<DESCRIPTION
DESCRIPTION
}

variable "subnet" {
  type = list(object({
    tier          = string
    address       = string
    security_zone = string
    network_security_rules = optional(map(object({
      access                                     = string
      name                                       = string
      description                                = optional(string)
      destination_address_prefix                 = optional(string)
      destination_address_prefixes               = optional(set(string))
      destination_application_security_group_ids = optional(set(string))
      destination_port_range                     = optional(string)
      destination_port_ranges                    = optional(set(string))
      direction                                  = string
      priority                                   = number
      protocol                                   = string
      source_address_prefix                      = optional(string)
      source_address_prefixes                    = optional(set(string))
      source_application_security_group_ids      = optional(set(string))
      source_port_range                          = optional(string)
      source_port_ranges                         = optional(set(string))
      timeouts = optional(object({
        create = optional(string)
        delete = optional(string)
        read   = optional(string)
        update = optional(string)
      }))
    })), {})
  }))
}

variable "wl_subscription_id" {
  type    = string
  default = "177c1f84-4b2a-414d-974e-5fbdf5d84250"
}

variable "role_assignments" {
  type = map(object({
    principal_id      = string
    definition        = string
    relative_scope    = optional(string, "")
    condition         = optional(string, "")
    condition_version = optional(string, "")
  }))
  default = {}
}

variable "rbac_info_principalid" {
  type    = map(list(string))
  default = {}
}

variable "rbac_info_groupname" {
  type    = map(list(string))
  default = {}
}
variable "rbac_info_mi" {
  type    = map(list(string))
  default = {}
}

variable "hub" {
  type = object({
    hub_rg_name     = string
    vnet_name       = string
    firewall_name   = string
    subscription_id = string
  })
}

variable "cmk" {
  description = "Configuration for Customer Managed Key"
  default = null
  type = object({
    subscription_id = string
    rg_name         = string
    keyvault_name   = string
    vm_key_name     = string  //VM DES Key in Key Vault
    st_key_name     = string  // Storage account CMK for encryption
    uai_name        = string
  })
}

variable "tenant_id" {
  type    = string
  default = "68283f3b-8487-4c86-adb3-a5228f18b893"
}

variable "storageaccount_ip_rules" {
  type = set(string)
  default = []
}

variable "is_oracle_subscription" {
  type        = bool
  description = "Flag to indicate if the subscription is for Oracle. If true, adds Oracle Database subnet delegation."
  default     = false
}

variable "az_backup_enable" {
  type = bool
  description = "Flag to determine of Azure backup should be enabled"
  default = true
}

variable "overwrite_route_subnets_list" {
  type = list
  description = "List of subnets from the Routable Hub that need to be added in the route table. Note: GatewaySubnet, AzureFirewallManagementSubnet, AzureFirewallSubnets are handled by default."
  default = []
}

variable "is_odaa_peering_enabled" {
  type        = bool
  description = "Flag to indicate if the subscription is for Oracle and require VNET peering with respective ODAA VNET"
  default     = false
}

variable "odaa_peering" {
 type = map(object({
   subscription_id = string
   odaa_vnet_name   = string
   odaa_rg_name     = string
 }))
 description = "Oracle VNET peering configuration for each environment (prod and nprd)"
 default = {}
}

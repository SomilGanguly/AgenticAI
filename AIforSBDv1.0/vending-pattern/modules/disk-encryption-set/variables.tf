variable "auto_key_rotation_enabled" {
  description = "Enable or disable automatic key rotation for the Disk Encryption Set."
  type        = bool
  default     = false  
}
 
variable "disk_encryption_set_name" {
  description = "The name of the Disk Encryption Set."
  type        = string
}
 
variable "resource_group_name" {
  description = "The resource group where the DES will be created."
  type        = string
}
 
variable "location" {
  description = "The location where the Disk Encryption Set will be created."
  type        = string
}
 
variable "uai_ids" {
  description = "The list of User Assigned Identity (UAI) IDs for the DES."
  type        = list(string)
}
 
variable "key_vault_key_id" {
  description = "The ID of the Key Vault key to use for encryption."
  type        = string
}
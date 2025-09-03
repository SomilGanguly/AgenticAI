output "disk_encryption_set_id" {
  description = "The ID of the created Disk Encryption Set."
  value       = azurerm_disk_encryption_set.this.id
}
-- Fix foreign key constraints to allow cascading deletes
-- Run this script in phpMyAdmin or MySQL Workbench

-- Drop existing foreign key constraint from payments table
ALTER TABLE `payments` 
DROP FOREIGN KEY `FKj94hgy9v5fw1munb90tar2eje`;

-- Add foreign key constraint with ON DELETE CASCADE
ALTER TABLE `payments`
ADD CONSTRAINT `fk_payments_user_id`
FOREIGN KEY (`user_id`) 
REFERENCES `users` (`id`)
ON DELETE CASCADE;

-- Check if ocr_history has a foreign key constraint and fix it too
-- First, find the constraint name (it might vary)
-- You can check with: SHOW CREATE TABLE ocr_history;

-- If ocr_history has a foreign key to users, drop and recreate it with CASCADE
-- Example (uncomment and adjust constraint name if needed):
-- ALTER TABLE `ocr_history` 
-- DROP FOREIGN KEY `FK_CONSTRAINT_NAME_HERE`;

-- ALTER TABLE `ocr_history`
-- ADD CONSTRAINT `fk_ocr_history_user_id`
-- FOREIGN KEY (`user_id`) 
-- REFERENCES `users` (`id`)
-- ON DELETE CASCADE;

-- Verify the changes
SHOW CREATE TABLE `payments`;
-- SHOW CREATE TABLE `ocr_history`;

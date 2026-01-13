-- Create contacts table for storing contact form submissions
CREATE TABLE IF NOT EXISTS `contacts` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL,
  `subject` VARCHAR(255) NOT NULL,
  `message` TEXT NOT NULL,
  `submitted_at` DATETIME NOT NULL,
  `read` BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create index on submitted_at for faster ordering
CREATE INDEX idx_submitted_at ON `contacts` (`submitted_at` DESC);

-- Create index on read status for quick filtering
CREATE INDEX idx_read ON `contacts` (`read`);

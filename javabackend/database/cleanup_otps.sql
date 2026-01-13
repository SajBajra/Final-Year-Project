-- Clean up OTP table for testing
-- This removes all OTPs to start fresh

DELETE FROM otps;

-- Or if you want to keep OTPs but reset verified flag:
-- UPDATE otps SET verified = FALSE WHERE verified = TRUE;

-- Check remaining OTPs:
-- SELECT * FROM otps;

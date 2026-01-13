# eSewa Payment Integration Setup Guide

This document provides instructions for setting up the eSewa payment gateway integration in the Lipika OCR application.

## Overview

The application integrates with [eSewa Payment Gateway](https://developer.esewa.com.np/) to allow users to upgrade to Premium plans with unlimited OCR scans.

## Backend Setup

### 1. Configuration File

The payment gateway configuration is stored in `javabackend/src/main/resources/application-payment.properties`.

**⚠️ IMPORTANT**: This file contains sensitive credentials and is **git-ignored** for security.

#### For Development/Testing:

The repository includes test credentials from eSewa's official documentation:
- Merchant Code: `EPAYTEST`
- Secret Key: `8gBm/:&EnhH.1/q`
- Test Environment URLs are pre-configured

#### For Production:

1. Contact eSewa to obtain live merchant credentials:
   - Email: merchant.operation@esewa.com.np
   - Website: https://esewa.com.np/

2. Update `application-payment.properties`:
   ```properties
   esewa.merchant.code=YOUR_LIVE_MERCHANT_CODE
   esewa.secret.key=YOUR_LIVE_SECRET_KEY
   esewa.payment.url=https://epay.esewa.com.np/api/epay/main/v2/form
   esewa.status.check.url=https://epay.esewa.com.np/api/epay/transaction/status/
   ```

3. Update callback URLs to match your production domain:
   ```properties
   esewa.success.url=https://yourdomain.com/payment/success
   esewa.failure.url=https://yourdomain.com/payment/failure
   ```

### 2. Database Setup

The payment integration requires a `payments` table. Ensure your database is updated with the Payment entity schema.

Run the application to auto-create the table (if using JPA auto-ddl), or manually create:

```sql
CREATE TABLE payments (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    transaction_uuid VARCHAR(255) UNIQUE NOT NULL,
    product_code VARCHAR(100),
    product_name VARCHAR(255),
    amount DOUBLE NOT NULL,
    tax_amount DOUBLE,
    service_charge DOUBLE,
    delivery_charge DOUBLE,
    total_amount DOUBLE NOT NULL,
    status VARCHAR(50) NOT NULL,
    esewa_ref_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    verified_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

## Frontend Setup

### Environment Variables

No additional environment variables are required. The frontend uses the existing `VITE_API_URL` to connect to the backend payment API.

### Payment Flow

1. User navigates to `/payment` page
2. Selects a premium plan
3. Clicks "Pay with eSewa"
4. Backend generates payment signature and returns payment parameters
5. Frontend submits a form to eSewa's payment gateway
6. User completes payment on eSewa
7. eSewa redirects to success/failure callback URL
8. Backend verifies payment with eSewa API
9. User sees confirmation page

## Testing

### Test with eSewa

1. Use the test credentials already configured
2. Navigate to `/payment` in your application
3. Select any plan
4. You will be redirected to eSewa's test environment
5. Use any test eSewa account to complete payment
6. Verify the payment is recorded in your database

### Test Credentials

The application is pre-configured with eSewa's official test credentials:
- Environment: RC (Release Candidate)
- Merchant Code: EPAYTEST
- Secret Key: 8gBm/:&EnhH.1/q

Reference: https://developer.esewa.com.np/pages/Test-credentials

## API Endpoints

### Initiate Payment
```
POST /api/payment/initiate
Authorization: Bearer <token>
Body: {
  "amount": 500,
  "productName": "Premium Monthly",
  "productCode": "PREMIUM_MONTHLY"
}
```

### Verify Payment
```
GET /api/payment/verify?data=<base64_encoded_data>
```

### Get Payment History
```
GET /api/payment/history
Authorization: Bearer <token>
```

### Get Payment Details
```
GET /api/payment/{transactionUuid}
```

## Security

### What's Protected

1. **Configuration File**: `application-payment.properties` is git-ignored
2. **Signature Generation**: Uses HMAC-SHA256 for secure payment signatures
3. **Payment Verification**: Double-checks with eSewa's API before confirming
4. **User Authentication**: Payment endpoints require JWT authentication

### Best Practices

1. Never commit `application-payment.properties` to version control
2. Use environment-specific credentials (test for dev, live for production)
3. Always verify payments server-side
4. Monitor payment logs for suspicious activities
5. Keep eSewa secret key secure and rotate it periodically

## Pricing Plans

Current premium plans configured:

| Plan | Price (NPR) | Duration | Product Code |
|------|-------------|----------|--------------|
| Premium Monthly | 500 | 30 days | PREMIUM_MONTHLY |
| Premium Yearly | 5,000 | 365 days | PREMIUM_YEARLY |

## Troubleshooting

### Common Issues

1. **Payment initiation fails**
   - Check if `application-payment.properties` exists
   - Verify merchant credentials are correct
   - Check backend logs for signature generation errors

2. **Payment verification fails**
   - Ensure callback URLs are correctly configured
   - Check if eSewa can reach your server (not behind firewall)
   - Verify the `data` parameter from eSewa is being passed correctly

3. **Database errors**
   - Ensure `payments` table exists
   - Check foreign key constraints with `users` table
   - Verify user is authenticated before payment

### Logs

Check these log files for payment-related issues:
- Backend: Look for `PaymentServiceImpl` and `PaymentController` logs
- Frontend: Check browser console for API errors

## Support

- **eSewa Technical Support**: merchant.operation@esewa.com.np
- **eSewa Documentation**: https://developer.esewa.com.np/
- **API Reference**: https://developer.esewa.com.np/pages/Epay

## Files Modified

### Backend
- `javabackend/src/main/java/com/lipika/config/EsewaConfig.java`
- `javabackend/src/main/java/com/lipika/controller/PaymentController.java`
- `javabackend/src/main/java/com/lipika/dto/PaymentRequest.java`
- `javabackend/src/main/java/com/lipika/dto/PaymentInitiateResponse.java`
- `javabackend/src/main/java/com/lipika/dto/PaymentVerificationResponse.java`
- `javabackend/src/main/java/com/lipika/model/Payment.java`
- `javabackend/src/main/java/com/lipika/repository/PaymentRepository.java`
- `javabackend/src/main/java/com/lipika/service/PaymentService.java`
- `javabackend/src/main/java/com/lipika/service/impl/PaymentServiceImpl.java`
- `javabackend/src/main/resources/application-payment.properties`

### Frontend
- `frontend/src/pages/Payment.jsx`
- `frontend/src/pages/PaymentSuccess.jsx`
- `frontend/src/pages/PaymentFailure.jsx`
- `frontend/src/services/paymentService.js`
- `frontend/src/pages/UserProfile.jsx`
- `frontend/src/App.jsx`

### Configuration
- `.gitignore` - Updated to exclude payment config files

## License

eSewa integration follows eSewa's terms and conditions. See https://esewa.com.np/ for details.

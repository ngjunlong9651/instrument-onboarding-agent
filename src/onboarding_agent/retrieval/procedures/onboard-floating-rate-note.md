---
id: onboard-floating-rate-note
title: Onboarding a floating rate note (FRN)
workflow: instrument_onboarding
steps: lookup_reference_data, check_booking_system, create_instrument
---
Floating rate notes pay a coupon that resets against a benchmark such as SOFR or SONIA plus a spread. Use this procedure for new FRN setups.

1. Pull reference data. The coupon field holds the current fixing; day count is usually ACT/360.
2. Confirm the FRN is not already booked.
3. Create the instrument record for approval. Reset schedules are maintained by the rates team.

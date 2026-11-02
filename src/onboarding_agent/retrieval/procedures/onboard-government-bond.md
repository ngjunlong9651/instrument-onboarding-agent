---
id: onboard-government-bond
title: Onboarding a new government bond
workflow: instrument_onboarding
steps: lookup_reference_data, check_booking_system, create_instrument
---
Use this procedure to set up a new sovereign or government bond (Treasuries, gilts, Bunds, OATs, SGS, ACGBs) in the booking system so the desk can trade it.

1. Pull static data for the ISIN from the reference data vendor: coupon, issue and maturity dates, currency, day count convention and issuer.
2. Check the booking system to confirm the security is not already set up. Duplicates break position keeping.
3. Create the instrument in the booking system. This is a production write and needs approval from market data operations.

Government bonds normally use ACT/ACT day count. Zero coupon issues are allowed.

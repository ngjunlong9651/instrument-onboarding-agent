---
id: price-validation-eod
title: End of day price validation
workflow: price_validation
steps: get_vendor_prices:primary, get_vendor_prices:secondary, compare_price_sources, publish_validated_price
---
Daily end of day (EOD) price check before prices are published to risk and valuation systems.

1. Get the clean price from the primary pricing vendor.
2. Get the clean price from the secondary vendor as an independent check.
3. Compare the two sources. They must agree within tolerance, be within a plausible range (percent of par) and be fresh.
4. Publish the validated price to the booking system for marking positions. Publishing needs approval.

Stale or out of range prices are never published.

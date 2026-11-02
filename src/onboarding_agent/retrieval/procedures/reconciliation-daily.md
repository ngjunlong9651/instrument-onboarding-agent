---
id: reconciliation-daily
title: Daily vendor feed reconciliation
workflow: vendor_reconciliation
steps: get_vendor_prices:primary, get_vendor_prices:secondary, compare_price_sources, flag_reconciliation_break
---
Reconcile the primary and secondary vendor price feeds every day and record any breaks for investigation.

1. Pull prices from both feeds.
2. Compare them. A difference above tolerance is a reconciliation break.
3. Raise a break ticket with both values and the difference so operations can investigate which feed is wrong.

If the feeds agree, the reconciliation is clean and no ticket is raised.

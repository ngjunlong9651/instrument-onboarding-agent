---
id: vendor-format-change-runbook
title: Runbook for a vendor field or format change
workflow: vendor_reconciliation
steps: escalate_to_human
---
When a vendor changes the format, unit or meaning of a field (for example quoting price as a fraction of par instead of percent, or changing a date format), automated checks will start failing for a whole feed.

Stop automated writes for the affected feed, escalate to market data operations with sample payloads, and contact the vendor. Resume only after the adapter mapping is updated and the feed passes validation again.

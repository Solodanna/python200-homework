# Project 08

Use this file for your Part 1 and Part 2 project write-up.

## Part 1: Cost Analysis Write-Up

- Objective:
  Estimate low-end and high-end monthly infrastructure costs for a data pipeline in East US (Linux), then compare cost drivers.

- Assumptions:
  Scenario A uses one Standard_B1s VM for 160 hours/month.
  Scenario B uses one Standard_NC6s_v3 VM for 730 hours/month, Azure SQL Database General Purpose (4 vCores) for 730 hours/month, and 1 TB of Blob Storage (Hot LRS) for one month.
  Pricing is pay-as-you-go retail pricing from Azure public pricing data and can vary by date, subscription discounts, taxes, or reserved/spot pricing choices.

- Method:
  I looked up East US prices and multiplied unit rates by usage.
  Scenario A formula: VM hourly rate _ 160.
  Scenario B formula: (GPU VM hourly rate _ 730) + (SQL hourly rate _ 730) + (Blob per-GB-month rate _ 1024 GB).

- Results:
  Scenario A (Standard_B1s Linux):
  Hourly rate: $0.0104
  Monthly estimate: 0.0104 \* 160 = $1.66

Scenario B (heavy analytics):
Standard_NC6s_v3 Linux VM hourly rate: $3.06
VM monthly estimate: 3.06 \* 730 = $2,233.80

Azure SQL Database (General Purpose, 4 vCores) hourly rate: $0.608868
SQL monthly estimate: 0.608868 \* 730 = $444.47

Azure Blob Storage Hot LRS data stored rate: $0.0208 per GB-month
Storage monthly estimate: 0.0208 \* 1024 = $21.30

Scenario B total estimate (VM + SQL + Blob): $2,699.57/month

- Conclusion:
  The biggest surprise was how dominant GPU compute is. The Scenario B GPU VM alone is about 1,343x the cost of Scenario A ($2,233.80 vs $1.66), and the full high-end stack is about 1,626x Scenario A. Storage at 1 TB was comparatively small in this setup, while always-on GPU and database compute drove almost all of the monthly cost.

## Part 2: Video Link

- Video URL:

## Notes

-

# Warmup 08

## Cloud Concepts

### Question 1

The core cloud model is pay-as-you-go: you rent compute, storage, and services when you need them and stop paying when you do not. That differs from owning servers, where you buy hardware up front, maintain it yourself, and pay for capacity even when it sits idle.

### Question 2

Vertical scaling means making one machine bigger (more CPU, RAM, or GPU), while horizontal scaling means adding more machines and spreading work across them. I would choose vertical scaling for a single heavy workload that does not split easily, and horizontal scaling for traffic or jobs that can run in parallel.

#### Scenario classification

- A web app that jumps from 1,000 to 100,000 users per day: horizontal scaling, because many app instances can share incoming requests.
- A model training job needing faster GPU and more RAM: vertical scaling, because the goal is a stronger single machine.
- A pipeline growing from 10 to 10,000 files with splittable work: horizontal scaling, because file processing can be distributed across multiple workers.

### Question 3

#### IaaS, PaaS, or SaaS classification

- Gmail: SaaS, because you consume a finished software product and do not manage infrastructure.
- Azure Virtual Machines: IaaS, because you rent raw compute infrastructure and manage the OS/runtime yourself.
- Azure App Service: PaaS, because you deploy application code while Azure manages most platform operations.
- AWS S3 (Simple Storage Service): IaaS, because it is foundational infrastructure-level storage you build solutions on top of.
- GitHub Codespaces: PaaS, because it provides a managed development environment platform without server management.
- Snowflake: SaaS, because it is a fully managed data platform you use as a complete product.

#### Definitions in my own words

- IaaS is rented infrastructure (compute/network/storage) where I still manage the OS, runtime, and app stack. Example: Azure Virtual Machines; I manage system updates, packages, runtime, and my code.
- PaaS is a managed application platform where I focus mainly on my code and app config while the provider handles most operational plumbing. Example: Azure App Service; I manage app code, environment settings, and deployment.
- SaaS is ready-to-use software delivered over the internet, so I mostly manage users, data, and usage settings instead of servers/platform. Example: Gmail; I manage accounts and content, not infrastructure.

### Question 4

A managed data platform like Databricks or Snowflake is a specialized cloud service for analytics/data engineering that hides most infrastructure setup and tuning. Compared with using Azure directly, you gain speed, convenience, and built-in tooling, but give up some low-level control and often accept more platform lock-in and pricing tradeoffs.

### Question 5

The cloud is usually not the best choice when you need strict on-prem/local control due to compliance, regulation, or latency constraints, and when long-term steady workloads make owned infrastructure clearly cheaper than pay-as-you-go cloud pricing.

## Azure Basics

### Question 1

An Azure subscription is the top-level billing and governance container, while a resource group is a smaller container that organizes related resources inside a subscription. In our class setup, the Code the Dream subscription is shared, and each student has their own resource group.

### Question 2

Ephemeral Cloud Shell means the session environment can reset, so anything not stored in persistent storage can disappear between sessions. In this course setup, persistence comes from attaching Cloud Shell storage (Azure Files) so your home files survive.

### Question 3

The private key stays on my machine and proves my identity, while the public key is the shareable half that gets placed on remote systems. Uploading only the public key is safe because it cannot be used to derive the private key.

### Question 4

`az account show --output table` returns the current signed-in account details:

```text
EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantDefaultDomain                TenantDisplayName    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ---------------------------------  -------------------  ------------------------------------
AzureCloud         0f040ddd-301f-4665-8677-7b21f129d605  True         CTD Nonprofit Sponsorship  Enabled  codethedreamcloud.onmicrosoft.com  Code the Dream Inc   0f040ddd-301f-4665-8677-7b21f129d605
```

The default subscription is CTD Nonprofit Sponsorship, and the account is enabled on AzureCloud.

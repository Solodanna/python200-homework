"""Warmup 09 exercises."""

# --- Azure Authentication ---

# Q1
# When you run a Python script locally that uses DefaultAzureCredential, it
# checks a chain of possible credential sources and uses the first one that
# works. In this assignment, the important local source is your Azure CLI login
# session. That means you must run `az login` first so the Azure CLI can store
# an authenticated account token on your machine. DefaultAzureCredential knows
# to use it because AzureCliCredential is one of the credential types included
# in its built-in chain. If environment variables or another earlier credential
# source are not configured, it will try the Azure CLI session and use that
# cached login.

# Q2
# A deployed pipeline running on an Azure VM, App Service, container app, or
# other Azure-hosted service should not depend on `az login` because there is no
# human sitting there to complete an interactive sign-in, and storing personal
# login state on production infrastructure would be fragile and insecure.
# Instead, the deployed workload typically uses a managed identity or a service
# principal. DefaultAzureCredential includes ManagedIdentityCredential in the
# same chain, so the exact same Python code can work without changes: locally it
# may authenticate through the Azure CLI session, while in Azure it can detect
# and use the managed identity made available by the hosting environment.

# Q3
# The two most likely causes are:
# 1. You are not actually signed in locally, or the Azure CLI session expired,
#    so none of the credential sources in DefaultAzureCredential can get a
#    usable token. I would diagnose that by running `az account show` or
#    `az login` in the terminal and checking whether the expected subscription
#    and account are active.
# 2. You are authenticated, but the account or identity does not have the right
#    Azure permissions for the resource you are trying to access. In that case,
#    authentication may succeed but token acquisition or the next resource call
#    can still fail with an authentication/authorization-related error. I would
#    diagnose that by reading the full exception message, confirming which
#    credential in the chain was attempted, and then checking role assignments,
#    subscription context, resource group access, and whether the target service
#    expects data-plane permissions such as Blob Data Contributor.

# --- Blob Storage ---

# Q1
# Azure Blob Storage has a simple three-level hierarchy: storage account,
# container, and blob. The storage account is the top-level Azure resource that
# owns all the storage services and settings. Inside the account, containers act
# like named buckets that organize related files. Inside each container, blobs
# are the actual stored objects such as CSV files, JSON files, images, or model
# outputs. A good analogy is a filing cabinet: the storage account is the whole
# cabinet, a container is one drawer in that cabinet, and each blob is an
# individual document stored in the drawer.

# Q2
# A REST API returns a JSON payload each hour. You need to store the raw
# responses for reprocessing later: I would use Blob Storage because the raw
# JSON responses are file-like objects that are best kept cheaply in their
# original form for replay or reprocessing.
#
# Your pipeline produces a table of 50 million customer transactions that your
# analytics team queries by date range and customer ID every day: I would use a
# relational database such as Azure SQL because the team needs fast repeated
# queries with filtering, indexing, and structured table operations.
#
# A computer vision model produces image embeddings as NumPy arrays. You need to
# save them between pipeline runs: I would usually use Blob Storage because the
# embeddings are serialized binary artifacts rather than rows that people need
# to query interactively with SQL every day.

# Q3
def list_container(container_client) -> None:
	"""Print each blob name and size in bytes from a container."""
	for blob in container_client.list_blobs():
		print(f"{blob.name}: {blob.size}")


# Q4
def upload_text(container_client, blob_name: str, text: str) -> None:
	"""Upload UTF-8 text to a blob, replacing any existing blob with that name."""
	data = text.encode("utf-8")
	container_client.upload_blob(name=blob_name, data=data, overwrite=True)

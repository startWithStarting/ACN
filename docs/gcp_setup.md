# Deploying ACN Training to Google Cloud Platform (Vertex AI)

This guide explains how to run your PPO training loop on Google Cloud using Vertex AI Custom Training Jobs. This is ideal for using your $300 Free Trial credits.

## Prerequisites

1.  **Google Cloud Project**: Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2.  **Billing**: Ensure your free trial is active and linked to the project.
3.  **gcloud CLI**: Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) on your local machine.
4.  **Docker**: Ensure Docker is installed and running locally.

## Setup Steps

### 1. Enable APIs
Run these commands in your terminal (PowerShell or Bash) to enable necessary services:
```bash
gcloud services enable artifactregistry.googleapis.com \
                       compute.googleapis.com \
                       containerregistry.googleapis.com \
                       aiplatform.googleapis.com
```

### 2. Create an Artifact Registry Repository
This is where your Docker image will be stored.
```bash
gcloud artifacts repositories create acn-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="ACN Docker repository"
```

### 3. Build and Push the Docker Image
Replace `YOUR_PROJECT_ID` with your actual GCP project ID.
```bash
# Set your project ID variable
$PROJECT_ID = "your-project-id-here" 

# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build (tagging it for the registry)
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/acn-repo/acn-agent:v1 .

# Push to Google Cloud
docker push us-central1-docker.pkg.dev/$PROJECT_ID/acn-repo/acn-agent:v1
```

## Running the Training Job

You can submit the job purely from the command line without managing any VMs yourself.

### Create a Storage Bucket
Vertex AI needs a place to save your training artifacts (the models).
```bash
# Create a unique bucket name
$BUCKET_NAME = "acn-training-results-" + (Get-Random)
gsutil mb -l us-central1 gs://$BUCKET_NAME
```

### Submit the Job
Use the `gcloud ai custom-jobs create` command. This spins up a specialized machine, runs your Docker container, saves the results to your bucket, and then shuts down (saving money).

**Example Command (CPU only):**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="acn-ppo-training-v1" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/$PROJECT_ID/acn-repo/acn-agent:v1 \
  --base-output-dir=gs://$BUCKET_NAME/job_output
```

**Example Command (GPU - T4):**
*Note: GPU usage might require a quota increase, which can be tricky on the Free Trial. Try CPU first or check "Vertex AI" quotas.*
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="acn-ppo-training-gpu" \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/$PROJECT_ID/acn-repo/acn-agent:v1 \
  --base-output-dir=gs://$BUCKET_NAME/job_output
```

## Syncing Results Back
After the job completes (check status in Console -> Vertex AI -> Training), download your trained models:
```bash
gsutil cp -r gs://$BUCKET_NAME/job_output/model/* ./results/models/
```

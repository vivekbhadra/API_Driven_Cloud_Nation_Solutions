import boto3
import datetime

def lambda_handler(event, context):
    glue = boto3.client('glue')
    s3 = boto3.client('s3')

    print("===== API Driven Cloud-Native Solution - Verification Output =====\n")

    # 1. Automatically discover all Glue jobs
    jobs_response = glue.get_jobs(MaxResults=20)
    job_names = [job['Name'] for job in jobs_response['Jobs']]
    print(f"Discovered {len(job_names)} Glue jobs in this account.\n")

    # 2. Retrieve 4 key application details for each job
    job_details = []
    for job in job_names:
        runs = glue.get_job_runs(JobName=job, MaxResults=1)
        if runs.get('JobRuns'):
            last_run = runs['JobRuns'][0]
            job_details.append({
                "Job Name": job,
                "Status": last_run.get('JobRunState', 'N/A'),
                "Started On": str(last_run.get('StartedOn', 'N/A')),
                "Execution Time (s)": last_run.get('ExecutionTime', 'N/A')
            })
        else:
            job_details.append({
                "Job Name": job,
                "Status": "No Runs Found",
                "Started On": "N/A",
                "Execution Time (s)": "N/A"
            })

    # 3. Print a clear verification table (4 key details)
    print("=== Verification Table: Four Application Details Retrieved via AWS APIs ===")
    print(f"{'Job Name':40s} {'Status':12s} {'Started On':28s} {'Exec Time (s)':>12s}")
    print("-" * 95)
    for d in job_details:
        print(f"{d['Job Name'][:38]:40s} {d['Status']:12s} {d['Started On'][:26]:28s} {str(d['Execution Time (s)']):>12s}")
    print("-" * 95)

    # 4. Fetch the most recent S3 artefacts for evidence
    bucket_name = "your-s3-bucket-name"  # Replace with actual bucket
    prefixes = ["models/", "processed/"]
    s3_results = {}
    for prefix in prefixes:
        try:
            r = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if "Contents" in r:
                latest = max(r["Contents"], key=lambda x: x["LastModified"])
                s3_results[prefix.strip("/")] = latest["Key"]
        except Exception as e:
            s3_results[prefix.strip("/")] = f"Error: {str(e)}"

    print("\n=== Additional Evidence: Latest S3 Artefacts ===")
    for k, v in s3_results.items():
        print(f"{k.capitalize():10s}: {v}")

    # 5. Return JSON for API Gateway / future use
    return {
        "statusCode": 200,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "glue_jobs": job_details,
        "s3_latest": s3_results
    }


from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import subprocess
import logging
import os
import sys

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    filename="microgrid_scheduler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# PIPELINE JOB
# --------------------------------------------------
def run_microgrid_pipeline():
    try:
        logging.info("Pipeline started")

        # Step 1: Forecast generation
        subprocess.run(
            [sys.executable, "7.py"],
            check=True
        )
        logging.info("Forecast generation completed")

        # Step 2: PPO inference
        subprocess.run(
            [sys.executable, "ppo_inference.py"],
            check=True
        )
        logging.info("PPO inference completed")

        logging.info("Pipeline finished successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")

# --------------------------------------------------
# SCHEDULER
# --------------------------------------------------
scheduler = BlockingScheduler()

scheduler.add_job(
    run_microgrid_pipeline,
    trigger="interval",
    minutes=5,
    max_instances=1,        # prevent overlapping runs
    coalesce=True           # merge missed runs
)

logging.info("Microgrid scheduler running (every 5 minutes)")
scheduler.start()

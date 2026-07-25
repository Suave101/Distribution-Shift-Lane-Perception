import os
import json
import uuid
import asyncio
from enum import Enum
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field, root_validator

app = FastAPI(
    title="Distribution Shift Detection API",
    description="API Wrapper for compiled Distribution-Shift-Lane-Perception binary",
    version="1.0.0",
)

# Path to the Nuitka compiled binary inside the container
BINARY_PATH = os.getenv("BINARY_PATH", "./shift_detector")

# In-memory job tracker (For production across restarts, consider Redis/SQL)
jobs_db: Dict[str, Dict[str, Any]] = {}


# =========================================================
# Schemas
# =========================================================
class WeightCommand(str, Enum):
    imagenet_weights = "imagenet_weights"
    random_weights = "random_weights"
    custom_weights = "custom_weights"


class ExperimentRequest(BaseModel):
    # Required arguments
    source_dir: str = Field(..., description="Absolute path to Source dataset directory")
    target_dir: str = Field(..., description="Absolute path to Target dataset directory")
    target_list_path: str = Field(..., description="Path to target list file")

    # Common optional arguments
    source_list_path: str = Field("./datasets/CULane/list/train.txt")
    sample_size: int = Field(1000, ge=1)
    num_runs: int = Field(100, ge=1)
    block_idx: int = Field(0, ge=0)
    batch_size: int = Field(128, ge=1)
    image_size: int = Field(512, ge=1)
    alpha: float = Field(0.05, gt=0, lt=1)
    seed_base: int = Field(42)
    permutation_test_iterations: int = Field(1000, ge=1)
    latent_dim: int = Field(32, ge=1)
    file_location: str = Field("logs", description="Directory to save the log JSON file")
    file_name: str = Field("experiment.json", description="Name of the log JSON file")

    # Data Perturbations
    gaussian_sigma: float = Field(0.0, ge=0)
    crop_image: bool = Field(False)
    rotation_angle: float = Field(0.0)
    width_shift_frac: float = Field(0.0)
    height_shift_frac: float = Field(0.0)
    shear_angle: float = Field(0.0)
    zoom_factor: float = Field(1.0)
    horizontal_flip: bool = Field(False)
    vertical_flip: bool = Field(False)

    # Subcommand & Custom Weights
    command: WeightCommand = Field(
        WeightCommand.imagenet_weights, description="Weight type subcommand"
    )
    model_weights_path: Optional[str] = Field(
        None, description="Path to directory containing custom weights (Required if command=custom_weights)"
    )

    @root_validator
    def validate_custom_weights(cls, values):
        command = values.get("command")
        weights_path = values.get("model_weights_path")
        if command == WeightCommand.custom_weights and not weights_path:
            raise ValueError("model_weights_path is required when command is set to 'custom_weights'")
        return values


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# =========================================================
# CLI Argument Builder
# =========================================================
def build_cli_args(req: ExperimentRequest) -> list[str]:
    """Converts the Pydantic request model into a CLI argument array for subprocess execution."""
    cmd = [
        BINARY_PATH,
        "--source_dir", req.source_dir,
        "--target_dir", req.target_dir,
        "--source_list_path", req.source_list_path,
        "--target_list_path", req.target_list_path,
        "--sample_size", str(req.sample_size),
        "--num_runs", str(req.num_runs),
        "--block_idx", str(req.block_idx),
        "--batch_size", str(req.batch_size),
        "--image_size", str(req.image_size),
        "--alpha", str(req.alpha),
        "--seed_base", str(req.seed_base),
        "--permutation_test_iterations", str(req.permutation_test_iterations),
        "--latent_dim", str(req.latent_dim),
        "--file_location", req.file_location,
        "--file_name", req.file_name,
        "--gaussian_sigma", str(req.gaussian_sigma),
        "--rotation_angle", str(req.rotation_angle),
        "--width_shift_frac", str(req.width_shift_frac),
        "--height_shift_frac", str(req.height_shift_frac),
        "--shear_angle", str(req.shear_angle),
        "--zoom_factor", str(req.zoom_factor),
    ]

    # Boolean flag parameters
    if req.crop_image:
        cmd.append("--crop_image")
    if req.horizontal_flip:
        cmd.append("--horizontal_flip")
    if req.vertical_flip:
        cmd.append("--vertical_flip")

    # Add Subparser command and its specific arguments
    cmd.append(req.command.value)
    if req.command == WeightCommand.custom_weights and req.model_weights_path:
        cmd.extend(["--model_weights_path", req.model_weights_path])

    return cmd


# =========================================================
# Async Background Worker
# =========================================================
async def run_experiment_task(job_id: str, req: ExperimentRequest):
    cmd = build_cli_args(req)
    jobs_db[job_id]["status"] = "running"
    jobs_db[job_id]["command_executed"] = " ".join(cmd)

    try:
        # Run binary process asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            jobs_db[job_id]["status"] = "completed"
            
            # Read generated result file if it exists
            result_file_path = os.path.join(req.file_location, req.file_name)
            if os.path.exists(result_file_path):
                with open(result_file_path, "r") as f:
                    jobs_db[job_id]["results"] = json.load(f)
            else:
                jobs_db[job_id]["results"] = {"notice": "Execution completed, but output JSON file was not found."}
        else:
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["error"] = stderr.decode("utf-8")

    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)


# =========================================================
# API Endpoints
# =========================================================
@app.post("/api/v1/experiments", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    """Submits a shift detection experiment job to run in the background."""
    job_id = str(uuid.uuid4())
    
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "request_params": req.dict(),
        "results": None,
        "error": None
    }

    # Queue execution in background
    background_tasks.add_task(run_experiment_task, job_id, req)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Experiment queued successfully. Poll GET /api/v1/experiments/{job_id} for results."
    )


@app.get("/api/v1/experiments/{job_id}")
async def get_experiment_status(job_id: str):
    """Retrieves current job status, execution logs, and experiment JSON results once finished."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    return jobs_db[job_id]


@app.get("/healthz")
async def health_check():
    """Kubernetes liveness/readiness health check probe endpoint."""
    return {"status": "ok", "binary_exists": os.path.exists(BINARY_PATH)}

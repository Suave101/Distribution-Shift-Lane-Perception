@echo off
setlocal enabledelayedexpansion

REM --- Job Configuration ---
REM This Windows batch script generates Slurm bash scripts for Linux HPC

echo ----------------------------------------------------
echo Job Name: bashGen
echo Running on host: %COMPUTERNAME%
echo Start Time: %date% %time%
echo ----------------------------------------------------

REM Parse command-line arguments with defaults
set K=10
set srcSamples=10
set SCRIPT_PREFIX=d128ids
set DCONFIG=d128ids
set OUTPUT_DIR=D:\Downloads\Distribution-Shift-Lane-Perception\LocalBash\IncreaseDimentionallity\ten128d

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--K" (
    set K=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--srcSamples" (
    set srcSamples=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--SCRIPT_PREFIX" (
    set SCRIPT_PREFIX=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--DCONFIG" (
    set DCONFIG=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--OUTPUT_DIR" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
shift
goto parse_args

:args_done

REM Validate required arguments
if "%K%"=="" (
    echo Error: K is required
    goto usage
)
if "%srcSamples%"=="" (
    echo Error: srcSamples is required
    goto usage
)
if "%SCRIPT_PREFIX%"=="" (
    echo Error: SCRIPT_PREFIX is required
    goto usage
)
if "%DCONFIG%"=="" (
    echo Error: DCONFIG is required
    goto usage
)
if "%OUTPUT_DIR%"=="" (
    echo Error: OUTPUT_DIR is required
    goto usage
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ========================================
echo Generating ratio experiment scripts...
echo Total samples (K): %K%
echo Source samples: %srcSamples%
echo Script prefix: %SCRIPT_PREFIX%
echo DCONFIG: %DCONFIG%
echo Output directory: %OUTPUT_DIR%
echo ========================================

REM Initialize experiment counter
set exp_num=1

REM Iterate from 0%% to 100%% source samples (in 10%% increments)
for /L %%p in (0,10,100) do (
    REM Calculate target percent (complement to 100%%)
    set /a tgt_percent=100-%%p
    
    REM Calculate actual sample counts
    set /a ratio_src=K*%%p/100
    set /a ratio_tgt=K*!tgt_percent!/100
    
    REM Define script filename
    set script_name=%OUTPUT_DIR%\!exp_num!%SCRIPT_PREFIX%.sh
    set log_name=!exp_num!%SCRIPT_PREFIX%.log
    set job_name=!exp_num!%SCRIPT_PREFIX%
    
    echo Creating Experiment !exp_num!: Src=%%p%% ^(!ratio_src!^), Tgt=!tgt_percent!%% ^(!ratio_tgt!^)
    
    REM Generate the Slurm bash script
    (
        echo #!/bin/bash
        echo.
        echo # --- Slurm Job Configuration ---
        echo #SBATCH --job-name=!job_name!
        echo #SBATCH --nodes=1
        echo #SBATCH --ntasks=1
        echo #SBATCH --output=!log_name!
        echo #SBATCH --partition=gpu2
        echo #SBATCH --gres=gpu:4
        echo #SBATCH --cpus-per-task=16
        echo #SBATCH --mem=128G
        echo #
        echo.    
        echo # --- Job Execution ---
        echo echo "----------------------------------------------------"
        echo echo "Slurm Job ID: $SLURM_JOB_ID"
        echo echo "Running on host: $(hostname)"
        echo echo "Start Time: $(date)"
        echo echo "Experiment !exp_num!: Src=%%p%% (!ratio_src! samples), Tgt=!tgt_percent!%% (!ratio_tgt! samples)"
        echo echo "----------------------------------------------------"
        echo.
        echo.
        echo export PYTHONNOUSERSITE=1
        echo.
        echo echo "Initializing conda for script..."
        echo source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
        echo.
        echo conda activate ml_project
        echo.
        echo echo "Conda environment activated and isolated:"
        echo conda info --env
        echo.
        echo cd /home1/adoyle2025/Distribution-Shift-Lane-Perception
        echo.
        echo echo "Starting Unit Test..."
        echo.
        echo python shift_concat_experiment.py \
        echo     --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
        echo     --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
        echo     --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
        echo     --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
        echo     --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
        echo     --src_samples %srcSamples% \
        echo     --tgt_samples %K% \
        echo     --ratio_src_samples !ratio_src! \
        echo     --ratio_tgt_samples !ratio_tgt! \
        echo     --num_runs 100 \
        echo     --block_idx 4 \
        echo     --seed_base 32 \
        echo     --batch_size 2048 \
        echo     --dConfig "%DCONFIG%" \
        echo     --save_all_image_paths True \
        echo     --file_name "%SCRIPT_PREFIX%.json"
        echo.
        echo echo "----------------------------------------------------"
        echo echo "Job finished: $(date)"
        echo echo "----------------------------------------------------"
    ) > "!script_name!"
    
    REM Increment experiment number
    set /a exp_num+=1
)

set /a total_scripts=exp_num-1

echo ========================================
echo Generated %total_scripts% scripts successfully!
echo.
echo Generated files:
dir /b "%OUTPUT_DIR%\%SCRIPT_PREFIX%*.sh"
echo.
echo To submit all jobs on the cluster:
echo   for script in *%SCRIPT_PREFIX%.sh; do sbatch $script; done
echo.
echo To submit individually:
for /L %%i in (1,1,%total_scripts%) do (
    echo   sbatch %%i%SCRIPT_PREFIX%.sh
)

goto end

:usage
echo.
echo Usage: bashGen.bat [OPTIONS]
echo.
echo Options:
echo   --K ^<value^>              Total samples (default: 10)
echo   --srcSamples ^<value^>     Source samples (default: 10)
echo   --SCRIPT_PREFIX ^<value^>  Script prefix (default: d128ids)
echo   --DCONFIG ^<value^>        Architecture config (default: d128ids)
echo   --OUTPUT_DIR ^<path^>      Output directory
echo.
echo Example:
echo   bashGen.bat --K 100 --srcSamples 100 --SCRIPT_PREFIX d128idsK100 --DCONFIG d128ids --OUTPUT_DIR D:\...\hundred128d
echo.
exit /b 1

:end
endlocal
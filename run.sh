#!/bin/bash

echo 'Submitting all SLURM jobs...'


# =========================================
# ImageNet Model Jobs
# =========================================
# sbatch LocalBash/Exodo/ImageNet/256d/10000sImageNet256d.sh
# sbatch LocalBash/Exodo/ImageNet/256d/1000sImageNet256d.sh
# sbatch LocalBash/Exodo/ImageNet/256d/100sImageNet256d.sh
sbatch LocalBash/Exodo/ImageNet/256d/10sImageNet256d.sh
# sbatch LocalBash/Exodo/ImageNet/128d/10000sImageNet128d.sh
# sbatch LocalBash/Exodo/ImageNet/128d/1000sImageNet128d.sh
sbatch LocalBash/Exodo/ImageNet/128d/100sImageNet128d.sh
sbatch LocalBash/Exodo/ImageNet/128d/10sImageNet128d.sh
# sbatch LocalBash/Exodo/ImageNet/32d/10000sImageNet32d.sh
# sbatch LocalBash/Exodo/ImageNet/32d/1000sImageNet32d.sh
sbatch LocalBash/Exodo/ImageNet/32d/100sImageNet32d.sh
sbatch LocalBash/Exodo/ImageNet/32d/10sImageNet32d.sh

# =========================================
# Random Model Jobs
# =========================================
# sbatch LocalBash/Exodo/Random/256d/10000sRandom256d.sh
sbatch LocalBash/Exodo/Random/256d/1000sRandom256d.sh
sbatch LocalBash/Exodo/Random/256d/100sRandom256d.sh
sbatch LocalBash/Exodo/Random/256d/10sRandom256d.sh
# sbatch LocalBash/Exodo/Random/128d/10000sRandom128d.sh
sbatch LocalBash/Exodo/Random/128d/1000sRandom128d.sh
sbatch LocalBash/Exodo/Random/128d/100sRandom128d.sh
sbatch LocalBash/Exodo/Random/128d/10sRandom128d.sh
# sbatch LocalBash/Exodo/Random/32d/10000sRandom32d.sh
sbatch LocalBash/Exodo/Random/32d/1000sRandom32d.sh
sbatch LocalBash/Exodo/Random/32d/100sRandom32d.sh
sbatch LocalBash/Exodo/Random/32d/10sRandom32d.sh

# =========================================
# CU_Lane Model Jobs
# =========================================
sbatch LocalBash/Exodo/CU_Lane/256d/10000sCU_Lane256d.sh
sbatch LocalBash/Exodo/CU_Lane/256d/1000sCU_Lane256d.sh
sbatch LocalBash/Exodo/CU_Lane/256d/100sCU_Lane256d.sh
sbatch LocalBash/Exodo/CU_Lane/256d/10sCU_Lane256d.sh
sbatch LocalBash/Exodo/CU_Lane/128d/10000sCU_Lane128d.sh
sbatch LocalBash/Exodo/CU_Lane/128d/1000sCU_Lane128d.sh
sbatch LocalBash/Exodo/CU_Lane/128d/100sCU_Lane128d.sh
sbatch LocalBash/Exodo/CU_Lane/128d/10sCU_Lane128d.sh
sbatch LocalBash/Exodo/CU_Lane/32d/10000sCU_Lane32d.sh
sbatch LocalBash/Exodo/CU_Lane/32d/1000sCU_Lane32d.sh
sbatch LocalBash/Exodo/CU_Lane/32d/100sCU_Lane32d.sh
sbatch LocalBash/Exodo/CU_Lane/32d/10sCU_Lane32d.sh

# =========================================
# CurveLanes Model Jobs
# =========================================
sbatch LocalBash/Exodo/CurveLanes/256d/10000sCurveLanes256d.sh
sbatch LocalBash/Exodo/CurveLanes/256d/1000sCurveLanes256d.sh
sbatch LocalBash/Exodo/CurveLanes/256d/100sCurveLanes256d.sh
sbatch LocalBash/Exodo/CurveLanes/256d/10sCurveLanes256d.sh
sbatch LocalBash/Exodo/CurveLanes/128d/10000sCurveLanes128d.sh
sbatch LocalBash/Exodo/CurveLanes/128d/1000sCurveLanes128d.sh
sbatch LocalBash/Exodo/CurveLanes/128d/100sCurveLanes128d.sh
sbatch LocalBash/Exodo/CurveLanes/128d/10sCurveLanes128d.sh
sbatch LocalBash/Exodo/CurveLanes/32d/10000sCurveLanes32d.sh
sbatch LocalBash/Exodo/CurveLanes/32d/1000sCurveLanes32d.sh
sbatch LocalBash/Exodo/CurveLanes/32d/100sCurveLanes32d.sh
sbatch LocalBash/Exodo/CurveLanes/32d/10sCurveLanes32d.sh

# =========================================
# ASSIST_Taxi Model Jobs
# =========================================
sbatch LocalBash/Exodo/ASSIST_Taxi/256d/10000sASSIST_Taxi256d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/256d/1000sASSIST_Taxi256d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/256d/100sASSIST_Taxi256d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/256d/10sASSIST_Taxi256d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/128d/10000sASSIST_Taxi128d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/128d/1000sASSIST_Taxi128d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/128d/100sASSIST_Taxi128d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/128d/10sASSIST_Taxi128d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/32d/10000sASSIST_Taxi32d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/32d/1000sASSIST_Taxi32d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/32d/100sASSIST_Taxi32d.sh
sbatch LocalBash/Exodo/ASSIST_Taxi/32d/10sASSIST_Taxi32d.sh

# =========================================
# DISTILL Model Jobs
# =========================================
sbatch LocalBash/Exodo/DISTILL/256d/10000sDISTILL256d.sh
sbatch LocalBash/Exodo/DISTILL/256d/1000sDISTILL256d.sh
sbatch LocalBash/Exodo/DISTILL/256d/100sDISTILL256d.sh
sbatch LocalBash/Exodo/DISTILL/256d/10sDISTILL256d.sh
sbatch LocalBash/Exodo/DISTILL/128d/10000sDISTILL128d.sh
sbatch LocalBash/Exodo/DISTILL/128d/1000sDISTILL128d.sh
sbatch LocalBash/Exodo/DISTILL/128d/100sDISTILL128d.sh
sbatch LocalBash/Exodo/DISTILL/128d/10sDISTILL128d.sh
sbatch LocalBash/Exodo/DISTILL/32d/10000sDISTILL32d.sh
sbatch LocalBash/Exodo/DISTILL/32d/1000sDISTILL32d.sh
sbatch LocalBash/Exodo/DISTILL/32d/100sDISTILL32d.sh
sbatch LocalBash/Exodo/DISTILL/32d/10sDISTILL32d.sh

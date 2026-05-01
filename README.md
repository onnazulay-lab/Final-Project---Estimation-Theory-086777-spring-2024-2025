# Belief-Space Autonomous Racing Project

## Overview
This repository contains a full two-agent autonomous racing simulation developed in **Julia**.

The project studies how two autonomous vehicles interact competitively while tracking uncertain beliefs about themselves, each other, and the environment.

---

## Repository Contents

### Core Simulation Code
The GitHub project includes the complete simulation source code for:

- Two racing agents (Blue / Red)
- Nonlinear vehicle motion on track
- EKF state estimation
- Belief covariance propagation
- Collision avoidance logic
- Track boundary constraints
- Obstacle / block interaction
- PCA-based covariance correction
- Differential-game inspired response logic
- Logging and data export

---

### Post-Processing Code
The repository also includes post-processing scripts used to generate:

- Publication-quality figures
- Comparative heatmaps
- Time-series plots
- Covariance mode evolution
- Constraint activation analysis
- Performance summaries

---

## Important Folders

### `timeseries_plots/`

Contains generated figures such as:

- vehicle trajectories  
- uncertainty ellipses  
- covariance traces  
- retained PCA rank heatmaps  
- control signals  
- belief evolution over time  

---

### `Testcases_Final/testcaseA1_Final/`

Contains exported simulation data for Case Study A1, including:

- CSV files of timeline variables  
- vehicle states vs time  
- covariance metrics  
- active modes history  
- retained-rank \(q_k\) history  
- constraint activity logs  
- performance metrics  

This folder is especially useful for reproducing figures or performing custom analysis.

---

## Requirements

Install:

- **Julia** (latest stable recommended)
- **VS Code**
- **Julia VS Code Extension**

---

## How to Enable Julia in VS Code

After installing Julia and the VS Code Julia extension:

### Open Julia REPL:

Press:

```text
Alt + j
then
Alt + o

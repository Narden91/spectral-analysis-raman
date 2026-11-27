# Roihu cheat sheet

CSC Roihu: 486 AMD Turin CPU nodes (384 cores each), 132 GH200 GPU nodes.
Docs: <https://docs.csc.fi> · Web UI: <https://roihu.csc.fi> · Projects:
<https://my.csc.fi>. Project: `project_2019570`.

Log in to `roihu-cpu` for CPU jobs and `roihu-gpu` for GPU jobs.

## One-time setup

```bash
# ~/.bashrc: plain sbatch/srun then need no --account/--partition
export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
export SALLOC_ACCOUNT=$SBATCH_ACCOUNT SRUN_ACCOUNT=$SBATCH_ACCOUNT
```

## Budget and storage

```bash
csc-projects                 # BU budget / used / remaining per project
csc-workspaces               # quota for home, projappl, scratch
seff <jobid>                 # efficiency + BU actually billed for a job
```

| Area | Quota | Use |
|---|---|---|
| `/users/$USER` | 15 GB | Config, small files |
| `/projappl/project_2019570` | 15 GB | Software, environments |
| `/scratch/project_2019570` | 250 GB | Data and results; **purged after 180 days** |

## Submit

```bash
sbatch job.sbatch                         # submit
sbatch --test-only job.sbatch             # validate + estimated start, no submit
sbatch --dependency=afterok:<id> b.sbatch # run after <id> succeeds
sbatch --array=1-10%4 job.sbatch          # array, max 4 at once
```

## Monitor

```bash
squeue --me                               # my jobs
squeue --me --start                       # estimated start times
scontrol show job <id>                    # full details, reason for pending
sacct -X -S today -o JobID,JobName%30,State,Elapsed,AllocCPUS,ReqMem
sstat -j <id>.batch -o AveCPU,MaxRSS      # live usage of a running job
tail -f path/to/log-<id>.out
sinfo -s                                  # partition load
```

## Control

```bash
scancel <id>                 # cancel one
scancel --me -n <job-name>   # cancel by name
scontrol hold <id>           # keep pending
scontrol release <id>
```

## Interactive

```bash
sinteractive -A project_2019570 -c 4 -m 16G -t 02:00:00   # shell; ≤32 cores, 64G, 36 h
srun -p test -c 8 --mem=16G -t 15 --pty bash   # quick test node, ≤15 min
```

## Partitions (CPU)

| Partition | Max time | Nodes | Notes |
|---|---|---|---|
| `test` | 15 min | 2 | Debugging |
| `interactive` | 36 h | 1 | ≤32 cores, 64G |
| `small` (default) | 3 days | 1 | Shared node |
| `longrun` | 10 days | 1 | ≤192 cores per user |
| `medium` | 36 h | 1–6 | Whole nodes |
| `large` | 36 h | 6–60 | Whole nodes |
| `hugemem` | 36 h | 1 | Up to ~6 TB, expensive |

## Billing (CPU BU per hour)

| Partition | Cost |
|---|---|
| `small`, `longrun`, `interactive`, `test` | max(0.75 × cores, 0.375 × GiB) |
| `medium`, `large` | 288 × nodes |
| `hugemem*` | max(12 × cores, 0.25 × GiB) |
| GPU partitions | 200 GPU BU × GPUs |

Keep memory at or below **2 GiB per core** on `small`, or memory sets the bill.
Billing uses elapsed time × *allocated* resources, not what you used.

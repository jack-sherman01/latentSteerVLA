# latentSteerVLA
steering VLA with latent space 

## Raw GR00T benchmark eval (ManiSkill / RoboCasa)

Before layering CompSteer's steering vectors on top, sanity-check that a
given GR00T checkpoint runs cleanly on each simulator via the raw (no
steering) eval scripts:

```bash
docker compose build

# ManiSkill — runs in-process, no server needed
docker compose run --rm groot-server \
    python scripts/eval_groot_maniskill.py --embodiment panda --tasks pick_cube

# RoboCasa — client/server: start the policy server, set up the isolated
# RoboCasa venv (one-time, ~10GB kitchen assets), then run the client
docker compose up -d groot-server-robocasa
docker compose run --rm robocasa-client scripts/setup_robocasa_env.sh
docker compose run --rm robocasa-client \
    python scripts/eval_groot_robocasa.py --tasks open_drawer --policy_host groot-server-robocasa
```

Results land in `results/raw_groot/{maniskill,robocasa}/`. Task/embodiment
registries live in `configs/tasks.yaml`, `configs/embodiments.yaml`, and
`configs/robocasa_tasks.yaml`.

Without Docker: `pip install -r requirements.txt`, then install
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and ManiSkill per their
own docs; RoboCasa needs its own venv (see `scripts/setup_robocasa_env.sh`)
due to a `robosuite` dependency conflict with GR00T.

See `slurm/README.md` for the full CompSteer (steering) pipeline.

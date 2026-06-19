import os


def add_wandb_args(parser, default_group="tacticsgpt"):
    parser.add_argument("--wandb_project", default="", help="Enable W&B logging by setting a project name.")
    parser.add_argument("--wandb_entity", default="", help="Optional W&B entity/team.")
    parser.add_argument("--wandb_run_name", default="", help="Optional W&B run name.")
    parser.add_argument("--wandb_group", default=default_group, help="Optional W&B group name.")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", default="", help="Comma-separated W&B tags.")


def init_wandb(args, stage, config):
    if not getattr(args, "wandb_project", ""):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested. Install it with `pip install wandb`.") from exc

    tags = [tag.strip() for tag in getattr(args, "wandb_tags", "").split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or "tacticsgpt",
        job_type=stage,
        mode=args.wandb_mode,
        tags=tags,
        config=config,
    )


def wandb_log(run, data, step=None):
    if run is not None:
        run.log(data, step=step)


def wandb_finish(run):
    if run is not None:
        run.finish()

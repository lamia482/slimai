import os


if __name__ == "__main__":
  print("Hello World")
  for k in [
    "LOCAL_RANK",
    "RANK", 
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
  ]:
    print(f"{k}: {os.environ[k]}")


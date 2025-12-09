import os
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
PATH_PREFIX = "/data/cyx" if os.path.exists("/data") else "/root"
local_rank = int(os.environ.get("LOCAL_RANK", 0))
SAVE_PATH = f"{PATH_PREFIX}/models/ADAK_MoE_{local_rank}"
print(SAVE_PATH)

"""Keep the mp4 of every failed episode + the first successful one of a run dir; delete the other mp4 (absolute paths from glob)."""
import glob, json, os, sys
d = os.path.abspath(sys.argv[1]); kept_ok = False
for j in sorted(glob.glob(d + "/episode_*.json")):
    mp4 = j[:-5] + ".mp4"
    if not os.path.exists(mp4):
        continue
    if json.load(open(j))["success"]:
        if kept_ok:
            os.remove(mp4)
        kept_ok = True

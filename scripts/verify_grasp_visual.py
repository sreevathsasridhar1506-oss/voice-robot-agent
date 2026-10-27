import sys
sys.path.insert(0, ".")

from sim_controller import SimController

sc = SimController(show=True)
sc.pick("block_red")
sc.save_snapshot("scripts/_snapshot_grasp_check.png")
print("saved")
print(sc.get_scene_state())
sc.close()

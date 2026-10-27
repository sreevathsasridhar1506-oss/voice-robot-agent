"""
One-off utility: capture a handful of clean simulation screenshots for the
portfolio site (not a verification script, doesn't assert anything).
"""
import sys
sys.path.insert(0, ".")

from sim_controller import SimController

OUT_DIR = "../portfolio_website/images"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    sc = SimController(show=False)
    try:
        sc.save_snapshot(f"{OUT_DIR}/sim-overview.png")
        print("captured overview")

        sc.pick("block_red")
        sc.save_snapshot(f"{OUT_DIR}/sim-grasp.png")
        print("captured grasp (holding red block)")

        sc.place([0.55, 0.0, 0.03])
        sc.save_snapshot(f"{OUT_DIR}/sim-place.png")
        print("captured place")

        sc.pick("block_blue")
        sc.move_to([0.45, -0.05, 0.35])
        sc.save_snapshot(f"{OUT_DIR}/sim-action.png")
        print("captured mid-motion action shot")
    finally:
        sc.close()


if __name__ == "__main__":
    main()

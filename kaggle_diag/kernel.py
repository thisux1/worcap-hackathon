import os
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count("/")
    if depth < 5:
        print(root, "| dirs:", dirs[:8], "| files:", files[:8])

import os

i = 0
for root, dirs, files in os.walk("./filtered_meshes"):
    if len(files) != 0:
        i = i + 1

print(i)

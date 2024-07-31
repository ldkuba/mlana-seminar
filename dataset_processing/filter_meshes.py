import os
import trimesh
import zipfile
import bpy
import numpy as np

output_dir = "./filtered_meshes_2/"

f = []
for (dirpath, dirnames, filenames) in os.walk("./"):
    f.extend(filter(lambda x: x.endswith(".zip"), filenames))
    break

decimated_count = 0

for file in f:
    with zipfile.ZipFile(file, "r") as f:
        for name in f.namelist():
            if name.endswith(".obj"):
                f.extract(name, path=output_dir)

                # Preprocess
                bpy.ops.wm.read_homefile(use_empty=True)
                bpy.ops.wm.obj_import(filepath=output_dir + name)
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.join()

                polygon_count = len(bpy.context.object.data.polygons)
                if polygon_count > 500:
                    # Try to decimate mesh
                    decimate_mod = bpy.context.object.modifiers.new(name="Decimate", type='DECIMATE')
                    decimate_mod.decimate_type = 'DISSOLVE'
                    decimate_mod.angle_limit = np.radians(15.0)

                    print("Polycount before:", len(bpy.context.object.data.polygons))
                    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)
                    print("Polycount after:", len(bpy.context.object.data.polygons))

                    polygon_count = len(bpy.context.object.data.polygons)
                    if polygon_count > 500:
                        print("Removing", polygon_count)
                        os.remove(output_dir + name)
                        folders = name.split("/")
                        if(len(os.listdir(output_dir + folders[0] + "/" + folders[1] + "/" + folders[2])) == 0):
                            os.rmdir(output_dir + folders[0] + "/" + folders[1] + "/" + folders[2])
                        if(len(os.listdir(output_dir + folders[0] + "/" + folders[1])) == 0):
                            os.rmdir(output_dir + folders[0] + "/" + folders[1])
                        if(len(os.listdir(output_dir + folders[0])) == 0):
                            os.rmdir(output_dir + folders[0])
                    else:
                        print("Keeping decimated", polygon_count)
                        decimated_count += 1
                        os.remove(output_dir + name)
                        bpy.ops.wm.obj_export(filepath=output_dir+name, check_existing=False, export_materials=False, export_uv=False, export_normals=False, export_triangulated_mesh=True)
        
                else:
                    print("Keeping", polygon_count)
                    os.remove(output_dir + name)
                    bpy.ops.wm.obj_export(filepath=output_dir+name, check_existing=False, export_materials=False, export_uv=False, export_normals=False, export_triangulated_mesh=True)

                print("Decimated count:", decimated_count)

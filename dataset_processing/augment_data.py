import os
import trimesh
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import bpy

for root, dirs, files in tqdm(os.walk("./filtered_meshes")):
    if len(files) > 0:
        filepath = root + "/model_normalized.obj"
        
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.wm.obj_import(filepath=filepath)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        bpy.ops.wm.obj_export(filepath=root+"/model_normalized.obj", check_existing=False, export_materials=False, export_uv=False, export_normals=False)

        # # Decimate mesh
        # decimate_mod = bpy.context.object.modifiers.new(name="Decimate", type='DECIMATE')
        # decimate_mod.decimate_type = 'DISSOLVE'
        # decimate_mod.angle_limit = np.radians(20)
        # bpy.ops.object.modifier_apply(modifier=decimate_mod.name)
        # bpy.ops.wm.obj_export(filepath=root+"/model_normalized_decimated.obj", check_existing=False, export_materials=False, export_uv=False, export_normals=False, export_triangulated_mesh=True)
        
        mesh = trimesh.load(filepath, force='mesh')
        # mesh_decimated = trimesh.load(root+"/model_normalized_decimated.obj", force='mesh')
        
        # # Calculate hausdorff distance
        # h_max = max(directed_hausdorff(mesh.vertices, mesh_decimated.vertices)[0], directed_hausdorff(mesh_decimated.vertices, mesh.vertices)[0])
        # print(h_max)

        # normalize mesh
        mesh.vertices -= mesh.centroid
        mesh.vertices /= max(mesh.bounds[1] - mesh.bounds[0])

        # Jitter vertices in [-0.1, 0.1] range
        mesh.vertices += np.random.uniform(-0.02, 0.02, mesh.vertices.shape)

        # Scale randomly in [0.75, 1.25] range
        scales = np.random.uniform(0.75, 1.25, (3))
        mesh.vertices *= scales

        # renormalize mesh
        mesh.vertices -= mesh.centroid
        mesh.vertices /= mesh.scale

        # Save augmented mesh
        augmented_filepath = filepath.replace("model_normalized.obj", "model_normalized_augmented.obj")
        if os.path.exists(augmented_filepath):
            os.remove(augmented_filepath)
        mesh.export(augmented_filepath)

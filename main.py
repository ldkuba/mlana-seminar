import torch
import trimesh
import meshgpt as mg
import os

if __name__ == "__main__":
    
    # Create dataset
    meshes = []
    for root, dirs, files in os.walk("../ShapeNetCore/filtered_meshes"):
        if len(files) > 0:
            for file in files:
                if file.endswith(".obj"):
                    meshes.append(root + "/" + file)

    dataset = mg.MeshDataset(meshes)

    # Create MeshGPTTrainer
    meshgpt = mg.MeshGPTTrainer(dataset)

    # Train autoencoder
    meshgpt.train_autoencoder("./saved_models/autoencoder_test", save_every=200, epochs=10)

    # # Train mesh transformer
    # meshgpt.train_mesh_transformer("./saved_models/mesh_transformer_{}.pth".format(datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")))

    # # Load autoencoder
    # meshgpt.load_autoencoder("./saved_models/autoencoder.pth")

    # # Load mesh transformer
    # meshgpt.load_mesh_transformer("./saved_models/mesh_transformer.pth")

    # # Generate mesh
    # generated_mesh = meshgpt.generate_mesh()
    # generated_mesh.show()
    # generated_mesh.export("./generate.obj")


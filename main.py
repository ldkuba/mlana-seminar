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

    if(not os.path.exists("./processed_dataset.pt")):
        dataset = mg.MeshDataset(meshes)
        dataset.save("./processed_dataset.pt")
    else:
        dataset = mg.MeshDataset()
        dataset.load("./processed_dataset.pt")


    # Create MeshGPTTrainer
    meshgpt = mg.MeshGPTTrainer(dataset)

    # Train autoencoder
    meshgpt.train_autoencoder("./saved_models/autoencoder_fixed_v2", save_every=200, epochs=40)

    # Train mesh transformer
    # meshgpt.load_autoencoder("./saved_models/autoencoder_v5_end.pth")

    # Load autoencoder
    # meshgpt.load_autoencoder("./saved_models/autoencoder_test_end.pth")

    # # Load mesh transformer
    # meshgpt.load_mesh_transformer("./saved_models/mesh_transformer.pth")

    # Generate mesh
    # generated_mesh = meshgpt.generate_mesh()
    # generated_mesh.show()
    # generated_mesh.export("./generate.obj")


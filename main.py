import torch
import trimesh
import meshgpt as mg
import os

if __name__ == "__main__":

    # Create dataset
    meshes = []
    recon_filename = ""
    for root, dirs, files in os.walk("../ShapeNetCore/filtered_meshes"):
        if len(files) > 0:
            for file in files:
                if file.endswith(".obj"):
                    meshes.extend([root + "/" + file] * 10)
                    recon_filename = root + "/" + file
                if len(meshes) >= 10:
                    break
        if len(meshes) >= 10:
            break

    if(not os.path.exists("./processed_dataset.pt")):
        dataset = mg.MeshDataset(meshes)
        dataset.save("./processed_dataset.pt")
    else:
        dataset = mg.MeshDataset()
        dataset.load("./processed_dataset.pt")


    # Create MeshGPTTrainer
    meshgpt = mg.MeshGPTTrainer(dataset)

    # Train autoencoder
    meshgpt.train_autoencoder(epochs=5000, batch_size=10, lr=1e-4, commit_weight=1000.0)
    reconstructed_mesh = meshgpt.reconstruct_mesh(recon_filename)
    reconstructed_mesh.show()

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


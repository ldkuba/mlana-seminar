import torch
import trimesh
import meshgpt as mg
import os

if __name__ == "__main__":
    
    t = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    o = torch.tensor([0.1, 0.2, 0.3])
    res = torch.matmul(t, o)

    # Create dataset
    meshes = []
    for root, dirs, files in os.walk("../ShapeNetCore/filtered_meshes"):
        if len(files) > 0:
            for file in files:
                if file.endswith(".obj"):
                    meshes.append(root + "/" + file)

    dataset = mg.MeshDataset(meshes[:2])

    # Create MeshGPTTrainer
    meshgpt = mg.MeshGPTTrainer(dataset)

    # Train autoencoder
    meshgpt.train_autoencoder(epochs=2)

    # # Train mesh transformer
    meshgpt.train_mesh_transformer(epochs=1, minimize_slivers=False)

    # Load autoencoder
    # meshgpt.load_autoencoder("./saved_models/autoencoder_test_end.pth")

    # # Load mesh transformer
    # meshgpt.load_mesh_transformer("./saved_models/mesh_transformer.pth")

    # Generate mesh
    generated_mesh = meshgpt.generate_mesh()
    generated_mesh.show()
    # generated_mesh.export("./generate.obj")


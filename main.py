import torch
import trimesh
import meshgpt as mg
import os

if __name__ == "__main__":

    # Dataset params
    dataset_name = '<saved-preprocessed-dataset-file-path>'
    dataste_location = '<dataset-root-dir-path>'
    
    # Wandb params
    wandb_entity = "<entity-name>"
    wandb_project = "<project-name>"
    wandb_run_name = "<run-name>"
    wandb_path = wandb_entity + "/" + wandb_project + "/" + wandb_run_name

    # Create dataset
    meshes = []
    for root, dirs, files in os.walk(dataste_location):
        if len(files) > 0:
            for file in files:
                if file.endswith(".obj"):
                    meshes.append(root + "/" + file)

    if not os.path.exists(dataset_name):
        dataset = mg.MeshDataset(meshes)
        dataset.save(dataset_name)
    else:
        dataset = mg.MeshDataset()
        dataset.load(dataset_name)

    # Create MeshGPTTrainer
    meshgpt = mg.MeshGPTTrainer()

    # === TRAIN AUTOENCODER ===
    
    # If resuming training, first load the model
    # meshgpt.load_autoencoder("./saved_models/autoencoder.pth")

    # meshgpt.train_autoencoder(
    #     dataset=dataset,
    #     epochs=3,
    #     save_every=1,
    #     batch_size=64,
    #     lr=1e-4,
    #     commit_weight=1.0
    # )
    # reconstructed_mesh = meshgpt.reconstruct_mesh([meshes[0]])
    # reconstructed_mesh.show()
    # reconstructed_mesh.export("./reconstructed.obj")

    # gt_mesh = trimesh.load(meshes[0], force='mesh')
    # gt_mesh.show()
    # gt_mesh.export("./gt.obj")

    # === TRAIN MESH TRANSFORMER ===
    # Load autoencoder
    # meshgpt.load_autoencoder("./saved_models/autoencoder.pth")

    # If resuming training, first load the model
    # meshgpt.load_mesh_transformer("./saved_models/mesh_transformer.pth")

    # Train mesh transformer
    # meshgpt.train_mesh_transformer(dataset=dataset, epochs=1, batch_size=1)

    # # Generate mesh
    # generated_mesh = meshgpt.generate_mesh()
    # generated_mesh.show()
    # generated_mesh.export("./generate.obj")

from VAE import save_model, train_base_model, train_conditional_vae, train_conditional_subspace_vae, train_discover_vae
import torch
import pandas as pd


if __name__ == "__main__":
    MODELS = ['Base', 'CVAE', 'CSVAE']  # Models to train
    MODEL_OUTPUT = 'models/large_models'

    # Load the raw data from CSV 
    full_dataset = pd.read_csv('data/chembl_35_fg_full.csv')
    
    torch.manual_seed(42)  # For reproducibility

    latent_dim = 16  # Example latent dimension
    encoder_hidden_dims = [1024, 512, 256, 128]  # Example encoder hidden layers
    decoder_hidden_dims = [128, 256, 1024]  # Example decoder hidden layers
    decoder_z_hidden_dims = [128, 256, 1024]  # Decoder layers for DISCoVeR
    latent_dim_z = 16 # for CSVAE
    latent_dim_w = 16 # for CSVAE
    encoder_hidden_dims_z = [1024, 512, 256, 128] # for CSVAE
    encoder_hidden_dims_w = [1024, 512, 256, 128] # for CSVAE
    adversarial_hidden_dims = [64] 
    batch_size = 64
    learning_rate = 1e-3
    max_epochs = 5

    fingerprint_dim = full_dataset['fp_length'].iloc[0]
    fg_dim = full_dataset['fg_length'].iloc[0]

    for MODEL in MODELS:
        if MODEL is None:
            raise ValueError("MODEL must be defined before training.")
        elif MODEL == 'Base':
            print("Training BaseVAE model...")

            vae_trainer = train_base_model(
                dataset=full_dataset,
                input_dim=fingerprint_dim,
                latent_dim=latent_dim,
                fg_dim=fg_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs
            )

            save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'CVAE':
            print("Training CVAE model...")

            vae_trainer = train_conditional_vae(
                dataset=full_dataset,
                fingerprint_dim=fingerprint_dim,
                fg_dim=fg_dim,
                latent_dim=latent_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs
            )

            save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'CSVAE':
                print("Training CSVAE model")

                vae_trainer = train_conditional_subspace_vae(
                    dataset=full_dataset,
                    fingerprint_dim=fingerprint_dim,
                    fg_dim=fg_dim,
                    latent_dim_z=latent_dim_z,
                    latent_dim_w=latent_dim_w,
                    encoder_hidden_dims_z=encoder_hidden_dims_z,
                    encoder_hidden_dims_w=encoder_hidden_dims_w,
                    decoder_hidden_dims=decoder_hidden_dims,
                    adversarial_hidden_dims=adversarial_hidden_dims,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs
                )
                save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'DISCoVeR':
                print("Training DISCoVeR VAE model...")

                vae_trainer = train_discover_vae(
                    dataset=full_dataset,
                    fingerprint_dim=fingerprint_dim,
                    fg_dim=fg_dim,
                    latent_dim_z=latent_dim_z,
                    latent_dim_w=latent_dim_w,
                    encoder_hidden_dims_z=encoder_hidden_dims_z,
                    encoder_hidden_dims_w=encoder_hidden_dims_w,
                    decoder_hidden_dims=decoder_hidden_dims,
                    decoder_z_hidden_dims=decoder_z_hidden_dims,
                    adversarial_hidden_dims=adversarial_hidden_dims,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs
                )
                save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        # Free memory after each model
        del vae_trainer
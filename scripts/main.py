import all the things

from models import fMRI_INRModel
from datasets import ImageINRDataset, ImageINRDatasetFourier
from data_utils import get_bw_array
from utils import train
#utils.py


###- MAIN TRAINING LOOP -###
def train(model, dataset, args):
    seed_everything()
    if args.model_type == 'gaussian':
        # init full dataset using Fourier features
        data = ImageINRDatasetFourier(df, gaussian=True, num_freq=num_freq, freq_file=freq_file, gauss_scale=gauss_scale)
        # init model
        model = fMRI_INRModel(input_size=4 * num_freq).to(device)
    else:
        print('Invalid model_type specified.')
        return
        

    #--- DEV SET SPLIT ---#

    # gets ds size (expected 9919)
    ds_size = len(data)
    # indexes length of ds
    indices = list(range(ds_size))

    # randomizes indices (keep seed for reproducibility)
    np.random.shuffle(indices)
    
    # calculates size of training set (80% of full set)
    train_split = int(np.floor(0.8 * ds_size))
    
    # calculates size of validation set (10% of full set)
    val_split = int(np.floor(0.9 * ds_size))

    # 0 - train_split .. (80%)
    train_indices = indices[:train_split]
    # train_split - val_split .. (10%)
    val_indices = indices[train_split:val_split]
    # val_split - END .. (10%)
    test_indices = indices[val_split:]

    # create subsets
    train_dataset = Subset(data, train_indices)
    val_dataset = Subset(data, val_indices)
    test_dataset = Subset(data, test_indices) # TEST - *not used in training

    print(f"DS SPLIT: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} testing.")

    # create dataloaders
    def seed_worker(worker_id):
        worker_seed = seed
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8192, shuffle=True, worker_init_fn=seed_worker, generator=g,)
    valid_dataloader = DataLoader(val_dataset, batch_size=8192, shuffle=False, worker_init_fn=seed_worker, generator=g,)

    
    #--- TRAINING ---#
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    patience = 30  
    epochs_no_improve = 0
    best_val_loss = float('inf')
    
    train_losses = []
    valid_losses = []

    num_epochs = 200

    for epoch in tqdm(range(num_epochs)):
        model.train()  
        
        running_train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

                 
        #--- VALIDATION ---#
        
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(valid_dataloader.dataset)
        valid_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }
            torch.save(checkpoint, model_file)
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
            
    # Save the losses for later analysis
    with open(f"{loss_file}_losses.pkl", "wb") as f:
        pickle.dump({'train': train_losses, 'validation': valid_losses}, f)


def main():
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    parser.add_argument("--dataset_path",default='/home/idies/workspace/Temporary/ecardillo/scratch/fmri/jubin-ds.nii')
    argparse 
    args
    # parameters for your experiment
    args.dataset_path = './nii'
    args.timepoint = 20
    args.z_idx = 10

    args.model_type = 'gaussian'
    args.num_freq = ~~
    args.dataset_type = '...'

    args.learning_rate = 1e-3
    args.num_epochs = 200
    args.patience = 30
    args.batch_size = 8192

    args.input_selection_method = 'nonzero'
    

    bw_array = 
    dataset = ~
    model = ~
    if args.model_type == 'gaussian':
        # init full dataset using Fourier features
        data = ImageINRDatasetFourier(df, gaussian=True, num_freq=args.num_freq, freq_file=args.freq_file, gauss_scale=args.gauss_scale)
        # init model
        model = fMRI_INRModel(input_size=4 * num_freq).to(device)
    else:
        print('Invalid model_type specified.')
        return
        
    train(model, dataset, args)
    
if __name__ == '__main__':
    main()
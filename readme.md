Demo code for neural encodec model, inspired by [EnCodec](https://github.com/facebookresearch/encodec). 

## Architecture
It's more or less the same as EnCodec, except I use SSIM instead of spectral distance (tried but does not work at all, maybe im wrong) and the discriminator is a Conv network, not MultiScaleSTFTDiscriminator (we have only 1 sample rate, and we want to reconstruct N channels, which can be thought as an image, i.e. 2D)
## Usage
Step 1: Download some Neuropixel data, raw. I used [this](https://rdr.ucl.ac.uk/articles/dataset/Recording_with_a_Neuropixels_probe/25232962/1).
Step 2: Activate env
```bash
conda env create -f environment.yml
```
Step 2: Create dataset
```bash
python data.py
```

Step 3: Train model
```bash
python enc.py
```
> :warning: **A lot of stuff is hardcoded**: Change accordingly!

Loss seems very high, but it's learning (other methods were worse). I guess also 1 hour of data is not much. 
```bash
Epoch 1/1000:  85%|████████▌ | 2156/2534 [15:52<02:46,  2.27it/s, batch=2157/2534, loss=534.8562, avg_loss=886.8040]
Epoch 3/1000:  25%|██▌       | 639/2534 [04:42<13:58,  2.26it/s, batch=639/2534, loss=455.6539, avg_loss=542.8949]
```

Step 4: Visualize reconstruction
```bash
# load model 
from enc import create_neural_encodec_model
import torch 
model = create_neural_encodec_model()
model.state_dict = torch.load('./best_model.pth')
model.eval()

#load data
data = torch.load('./fixed_dataset/segment_0001.pt')
dd = data.unsqueeze(0)
data2, _, _, _ = model(dd)
data2 = data2.detach().cpu().squeeze(0)

#plot original, reconstructed and difference
import numpy as np
import matplotlib.pyplot as plt


def plot_neural_data(data, data2, sample_rate=30000, num_electrodes_to_plot=20, time_window=0.03):
    data = data.cpu().numpy()
    data2 = data2.cpu().numpy()
    # Normalize the data
    def normalize(x):
        return (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-8)

    data_norm = normalize(data)
    data2_norm = normalize(data2)
    diff_norm = data2_norm - data_norm

    # Calculate the time axis
    time = np.arange(data.shape[1]) / sample_rate

    # Create the plot
    fig, axs = plt.subplots(3, 1, figsize=(20, 24), sharex=True)

    electrodes_to_plot = np.linspace(0, data.shape[0]-1, num_electrodes_to_plot, dtype=int)
    time_slice = slice(0, int(time_window * sample_rate))

    titles = ['Original Data (Normalized)', 'Reconstructed Data (Normalized)', 'Difference of Normalized Data']
    datasets = [data_norm, data2_norm, diff_norm]

    for idx, (ax, title, dat) in enumerate(zip(axs, titles, datasets)):
        for i, electrode in enumerate(electrodes_to_plot):
            ax.plot(time[time_slice], dat[electrode, time_slice] + i*4, 
                    linewidth=0.5, color='black', alpha=0.7)
        
        ax.set_title(title)
        ax.set_ylabel('Electrode Number')
        ax.set_ylim(-2, num_electrodes_to_plot * 4)
        ax.set_yticks(np.arange(0, num_electrodes_to_plot * 4, 20))
        ax.set_yticklabels(electrodes_to_plot[::5])
        
        # Add a scale bar (2 standard deviations)
        ax.plot([0.9*time_window, 0.9*time_window], [-1, 1], 'k', linewidth=2)
        ax.text(0.91*time_window, 0, '2σ', verticalalignment='center')

    axs[-1].set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.show()

plot_neural_data(data, data2)
plot_neural_data(data.T, data2.T) # this weirdly gives more stable results as far as I've trained it
```


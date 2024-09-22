import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedShiftNeuralDataset(Dataset):
    def __init__(self, file_path: str, channel_map_path: str, segment_length: int, num_segments: int):
        logger.info("Loading data...")
        self.dat = np.memmap(file_path, dtype=np.int16, mode='r')
        self.dat = self.dat.reshape((385, -1), order='F')
        
        logger.info("Applying channel map...")
        self.chanMap = np.load(channel_map_path)
        self.dat = self.dat[self.chanMap, :]
        self.dat = self.dat.squeeze(1)
        logger.info(f"Data shape: {self.dat.shape}")
        
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.shift = segment_length // 3 
        
        total_samples = self.dat.shape[1]
        
        logger.info("Generating segment indices...")
        self.segment_indices = []
        current_start = 0
        
        while len(self.segment_indices) < num_segments:
            if current_start + segment_length <= total_samples:
                # If we have enough data, use the next segment
                self.segment_indices.append(current_start)
                current_start += segment_length
            else:
                # If we don't have enough data, shift the last segment
                last_start = self.segment_indices[-1]
                new_start = last_start + self.shift
                if new_start + segment_length > total_samples:
                    # If shifting would go beyond the end, wrap around to the beginning
                    new_start = 0
                self.segment_indices.append(new_start)
            
            if len(self.segment_indices) % 1000 == 0:
                logger.info(f"Generated {len(self.segment_indices)} segment indices")
        
        logger.info(f"Total segments generated: {len(self.segment_indices)}")
        
    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = self.segment_indices[idx]
        end = start + self.segment_length
        
        if end <= self.dat.shape[1]:
            segment = self.dat[:, start:end]
        else:
            # Handle wrapping around the end of the data
            first_part = self.dat[:, start:]
            second_part = self.dat[:, :end - self.dat.shape[1]]
            segment = np.concatenate([first_part, second_part], axis=1)
        
        return torch.from_numpy(segment.astype(np.float32))

def create_and_save_dataset(file_path, channel_map_path, segment_length, num_segments, output_dir):
    dataset = FixedShiftNeuralDataset(file_path, channel_map_path, segment_length, num_segments)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving {num_segments} segments to {output_dir}...")
    for i in range(len(dataset)):
        segment = dataset[i]
        torch.save(segment, os.path.join(output_dir, f"segment_{i:04d}.pt"))
        if (i + 1) % 100 == 0:
            logger.info(f"Saved {i + 1} segments")
    
    logger.info("Dataset creation complete!")

if __name__ == "__main__":
    file_path = '/home/marisbasha/neural_encodec/original/Hopkins_20160722_g0_t0.imec.lf.bin'
    channel_map_path = '/home/marisbasha/neural_encodec/original/channel_map.npy'
    segment_length = 500
    num_segments = 64000
    output_dir = '/home/marisbasha/neural_encodec/fixed_shift_dataset'
    
    create_and_save_dataset(file_path, channel_map_path, segment_length, num_segments, output_dir)

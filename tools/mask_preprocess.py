import os
import glob
import numpy as np
import nibabel as nib
import scipy.io
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskPreprocessor:
    """Handle preprocessing of NIfTI mask files."""
    
    def __init__(self, target_size: int = 128):
        """Initialize the preprocessor.
        
        Args:
            target_size: Target size for all dimensions (default: 128)
        """
        self.target_size = target_size
        
    def read_nii(self, nii_filepath: str) -> np.ndarray:
        """Read NIfTI file and extract non-zero region with pixel dimension adjustment.
        
        Args:
            nii_filepath: Path to the NIfTI file
            
        Returns:
            Processed 3D mask array
        """
        try:
            # Load NIfTI file
            nii_img = nib.load(nii_filepath)
            mask3d = nii_img.get_fdata()
            header = nii_img.header
            
            # Get pixel dimensions
            pixel_dims = header.get_zooms()[:3]  # Get only spatial dimensions
            logger.info(f"Original pixel dimensions: {pixel_dims}")
            
            # Extract non-zero region (equivalent to MATLAB's find and ind2sub)
            nonzero_indices = np.nonzero(mask3d)
            
            if len(nonzero_indices[0]) == 0:
                logger.warning(f"No non-zero voxels found in {nii_filepath}")
                return np.zeros((1, 1, 1), dtype=bool)
            
            # Find bounding box of non-zero region
            min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
            min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
            min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])
            
            # Crop to non-zero region
            mask3d = mask3d[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
            logger.info(f"Cropped size: {mask3d.shape}")
            
            # Calculate target size based on pixel dimensions
            current_size = np.array(mask3d.shape)
            target_size = np.round(current_size * np.array(pixel_dims)).astype(int)
            logger.info(f"Target size after pixel dimension adjustment: {target_size}")
            
            # Resize using nearest neighbor interpolation
            if not np.array_equal(current_size, target_size):
                # Use zoom for 3D resizing with nearest neighbor
                zoom_factors = target_size / current_size
                mask3d_resized = zoom(mask3d, zoom_factors, order=0)  # order=0 for nearest neighbor
            else:
                mask3d_resized = mask3d
            
            # Convert to boolean
            mask3d_resized = mask3d_resized > 0
            
            return mask3d_resized.astype(bool)
            
        except Exception as e:
            logger.error(f"Error reading {nii_filepath}: {str(e)}")
            raise
    
    def scale_and_pad_mask(self, mask3d: np.ndarray) -> np.ndarray:
        """Scale mask to fit within target size and pad to exact target dimensions.
        
        Args:
            mask3d: Input 3D mask array
            
        Returns:
            Processed mask with exact target dimensions
        """
        original_size = np.array(mask3d.shape)
        logger.info(f"Original size: {original_size}")
        
        # Calculate scale factors to fit within target size
        scale_factors = np.min(self.target_size / original_size)
        target_size = np.round(original_size * scale_factors).astype(int)
        logger.info(f"Scaled target size: {target_size}")
        
        # Resize using nearest neighbor interpolation
        if not np.array_equal(original_size, target_size):
            zoom_factors = target_size / original_size
            scaled_mask = zoom(mask3d.astype(float), zoom_factors, order=0)
            scaled_mask = scaled_mask > 0.5  # Convert back to boolean
        else:
            scaled_mask = mask3d
        
        # Calculate padding needed
        padding_size = (self.target_size - target_size) / 2
        pad_before = np.floor(padding_size).astype(int)
        pad_after = np.ceil(padding_size).astype(int)
        
        logger.info(f"Padding before: {pad_before}, after: {pad_after}")
        
        # Apply padding
        padded_mask = np.pad(scaled_mask, 
                           [(pad_before[i], pad_after[i]) for i in range(3)], 
                           mode='constant', constant_values=0)
        
        # Ensure exact target size (handle any rounding errors)
        final_shape = padded_mask.shape
        if not np.array_equal(final_shape, [self.target_size] * 3):
            # Crop or pad to exact size if needed
            diff = np.array(final_shape) - self.target_size
            start_idx = np.maximum(0, diff // 2)
            end_idx = start_idx + self.target_size
            
            # Handle cases where final_shape might be smaller than target_size
            end_idx = np.minimum(end_idx, final_shape)
            actual_size = end_idx - start_idx
            
            cropped_mask = padded_mask[start_idx[0]:end_idx[0], 
                                     start_idx[1]:end_idx[1], 
                                     start_idx[2]:end_idx[2]]
            
            # If still not exact size, pad again
            if not np.array_equal(cropped_mask.shape, [self.target_size] * 3):
                remaining_pad = self.target_size - np.array(cropped_mask.shape)
                pad_final = [(0, remaining_pad[i]) for i in range(3)]
                padded_mask = np.pad(cropped_mask, pad_final, mode='constant', constant_values=0)
            else:
                padded_mask = cropped_mask
        
        logger.info(f"Final mask shape: {padded_mask.shape}")
        return padded_mask.astype(bool)
    
    def apply_transformations(self, mask: np.ndarray) -> np.ndarray:
        """Apply rotation and flipping transformations.
        
        Args:
            mask: Input 3D mask array
            
        Returns:
            Transformed mask
        """
        # Rotate 90 degrees counterclockwise 3 times (equivalent to -3 in MATLAB rot90)
        # In numpy, we use rot90 with k=-3 for the same effect
        rotated_mask = np.rot90(mask, k=-3, axes=(0, 1))
        
        # Flip along y-axis (second dimension)
        flipped_mask = np.flip(rotated_mask, axis=1)
        
        return flipped_mask
    
    def process_single_file(self, nii_filepath: str, save_path: str) -> bool:
        """Process a single NIfTI file.
        
        Args:
            nii_filepath: Path to input NIfTI file
            save_path: Directory to save output .mat file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing: {nii_filepath}")
            
            # Read and preprocess the NIfTI file
            mask3d = self.read_nii(nii_filepath)
            
            # Scale and pad to target size
            mask = self.scale_and_pad_mask(mask3d)
            
            # Apply transformations
            mask = self.apply_transformations(mask)
            
            # Generate output filename
            base_name = os.path.basename(nii_filepath)
            mat_filename = base_name.replace('.nii.gz', '.mat')
            mat_filepath = os.path.join(save_path, mat_filename)
            
            # Save as .mat file
            scipy.io.savemat(mat_filepath, {'mask': mask})
            logger.info(f"Saved: {mat_filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {nii_filepath}: {str(e)}")
            return False
    
    def process_folder(self, folder_path: str, save_path: str) -> None:
        """Process all NIfTI files in a folder.
        
        Args:
            folder_path: Input folder containing .nii.gz files
            save_path: Output folder for .mat files
        """
        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)
        
        # Find all .nii.gz files
        pattern = os.path.join(folder_path, '*.nii.gz')
        nii_files = glob.glob(pattern)
        
        if not nii_files:
            logger.warning(f"No .nii.gz files found in {folder_path}")
            return
        
        logger.info(f"Found {len(nii_files)} files to process")
        
        # Process each file
        successful = 0
        for i, nii_file in enumerate(nii_files, 1):
            logger.info(f"Processing file {i}/{len(nii_files)}")
            if self.process_single_file(nii_file, save_path):
                successful += 1
        
        logger.info(f"Processing complete: {successful}/{len(nii_files)} files processed successfully")


def main():
    """Main function to run the mask preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess NIfTI masks (.nii.gz) into cubic .mat files with specified target size."
    )
    parser.add_argument(
        "-i", "--input", required=False,
        help="Path to a .nii.gz file or a directory containing .nii.gz files. If omitted, falls back to script defaults."
    )
    parser.add_argument(
        "-o", "--output", required=False,
        help="Directory to save output .mat files. Defaults to input directory."
    )
    parser.add_argument(
        "-s", "--size", type=int, default=128,
        help="Target cubic size for output masks (e.g., 128). Default: 128"
    )
    args = parser.parse_args()

    if args.input is None:
        # Fallback to previous default configuration
        input_path = r'E:\phdplat\Data\WORD-V0.1.0\extract_labels'
        output_dir = args.output or r'E:\phdplat\Data\WORD-V0.1.0\extract_labels'
        logger.info("No --input provided. Falling back to script default paths.")
    else:
        input_path = args.input
        # Default output directory to input location (file's parent dir or the directory itself)
        output_dir = args.output or (input_path if os.path.isdir(input_path) else os.path.dirname(input_path))

    # Create preprocessor with desired target size
    preprocessor = MaskPreprocessor(target_size=args.size)

    # Branch based on whether input is a directory or a file
    if os.path.isdir(input_path):
        logger.info(f"Processing all .nii.gz files under directory: {input_path}")
        preprocessor.process_folder(input_path, output_dir)
    elif os.path.isfile(input_path):
        logger.info(f"Processing single file: {input_path}")
        os.makedirs(output_dir, exist_ok=True)
        ok = preprocessor.process_single_file(input_path, output_dir)
        if not ok:
            logger.error("Processing failed for the specified file.")
            raise SystemExit(1)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
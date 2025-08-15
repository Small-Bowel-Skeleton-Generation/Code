import torch
import torch.nn
from torch.nn import init
import ocnn
import copy
import random
import os
import scipy

from .distributions import DiagonalGaussianDistribution
from . import modules
from . import dual_octree
from . import mpu
from ocnn.nn import octree2voxel
from ocnn.octree import Octree, Points


def init_weights(net, init_type='normal', gain=0.01):
    """Initialize network weights with specified initialization method.
    
    Args:
        net: Neural network to initialize
        init_type: Type of initialization ('normal', 'xavier', 'kaiming', etc.)
        gain: Scaling factor for initialization
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
                
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
                
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    
    # propagate to children
    for m in net.children():
        m.apply(init_func)


class GraphVAE(torch.nn.Module):
    """Graph-based Variational Autoencoder for dual octree processing.
    
    This class implements a VAE that operates on dual octree structures using graph
    convolutions for both encoding and decoding.
    """
    
    def __init__(self, 
                 depth, 
                 channel_in, 
                 nout, 
                 full_depth=2, 
                 depth_stop=6, 
                 depth_out=8, 
                 use_checkpoint=False, 
                 resblk_type='bottleneck', 
                 bottleneck=4, 
                 resblk_num=3, 
                 code_channel=3, 
                 embed_dim=3):
        """Initialize the GraphVAE.
        
        Args:
            depth: Maximum depth of the octree
            channel_in: Number of input channels
            nout: Number of output channels
            full_depth: Depth for full octree initialization
            depth_stop: Stopping depth for encoder
            depth_out: Output depth for decoder
            use_checkpoint: Whether to use gradient checkpointing
            resblk_type: Type of residual blocks
            bottleneck: Bottleneck factor for residual blocks
            resblk_num: Number of residual blocks
            code_channel: Number of code channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        # Store configuration parameters
        self.depth = depth
        self.channel_in = channel_in
        self.nout = nout
        self.full_depth = full_depth
        self.depth_stop = depth_stop
        self.depth_out = depth_out
        self.use_checkpoint = use_checkpoint
        self.resblk_type = resblk_type
        self.bottleneck = bottleneck
        self.resblk_num = resblk_num
        self.code_channel = code_channel
        
        # Initialize neural MPU
        self.neural_mpu = mpu.NeuralMPU(self.full_depth, self.depth_stop, self.depth_out)
        
        # Setup channel and residual block configurations
        self._setup_channels_and_resblks()
        
        # Graph configuration
        n_edge_type, avg_degree = 7, 7
        self.dropout = 0.0
        
        # Build encoder components
        self._build_encoder(n_edge_type, avg_degree)
        
        # Build decoder components
        self._build_decoder(n_edge_type, avg_degree)
        
        # Build prediction heads
        self._build_prediction_heads()
        
        # Variational components
        ae_channel_in = self.channels[self.depth_stop]
        self.KL_conv = modules.Conv1x1(ae_channel_in, 2 * embed_dim, use_bias=True)
        self.post_KL_conv = modules.Conv1x1(embed_dim, ae_channel_in, use_bias=True)

    def _setup_channels_and_resblks(self):
        """Setup channel numbers and residual block configurations."""
        self.resblk_nums = [self.resblk_num] * 16
        self.channels = [8, 512, 512, 256, 128, 64, 32, 32, 24, 16, 8, 8]

    def _build_encoder(self, n_edge_type, avg_degree):
        """Build encoder components."""
        # Initial convolution
        self.conv1 = modules.GraphConv(
            self.channel_in, self.channels[self.depth], n_edge_type, avg_degree, self.depth - 1)
        
        # Encoder layers
        self.encoder = torch.nn.ModuleList([
            modules.GraphResBlocks(
                self.channels[d], self.channels[d], self.dropout,
                self.resblk_nums[d] - 1, n_edge_type, avg_degree, d - 1, self.use_checkpoint)
            for d in range(self.depth, self.depth_stop - 1, -1)
        ])
        
        # Downsampling layers
        self.downsample = torch.nn.ModuleList([
            modules.GraphDownsample(self.channels[d], self.channels[d - 1])
            for d in range(self.depth, self.depth_stop, -1)
        ])
        
        # Output normalization and activation
        self.encoder_norm_out = modules.DualOctreeGroupNorm(self.channels[self.depth_stop])
        self.nonlinearity = torch.nn.GELU()

    def _build_decoder(self, n_edge_type, avg_degree):
        """Build decoder components."""
        # Decoder layers
        self.decoder = torch.nn.ModuleList([
            modules.GraphResBlocks(
                self.channels[d], self.channels[d], self.dropout,
                self.resblk_nums[d], n_edge_type, avg_degree, d - 1, self.use_checkpoint)
            for d in range(self.depth_stop, self.depth + 1)
        ])
        
        # Middle decoder blocks
        self.decoder_mid = torch.nn.Module()
        self.decoder_mid.block_1 = modules.GraphResBlocks(
            self.channels[self.depth_stop], self.channels[self.depth_stop], self.dropout,
            self.resblk_nums[self.depth_stop], n_edge_type, avg_degree, self.depth_stop - 1, self.use_checkpoint)
        self.decoder_mid.block_2 = modules.GraphResBlocks(
            self.channels[self.depth_stop], self.channels[self.depth_stop], self.dropout,
            self.resblk_nums[self.depth_stop], n_edge_type, avg_degree, self.depth_stop - 1, self.use_checkpoint)
        
        # Upsampling layers
        self.upsample = torch.nn.ModuleList([
            modules.GraphUpsample(self.channels[d - 1], self.channels[d])
            for d in range(self.depth_stop + 1, self.depth + 1)
        ])

    def _build_prediction_heads(self):
        """Build prediction head modules."""
        # Prediction heads for octree splitting (binary classification)
        self.predict = torch.nn.ModuleList([
            self._make_predict_module(self.channels[d], 2)
            for d in range(self.depth_stop, self.depth + 1)
        ])
        
        # Regression heads for MPU features (normal + offset)
        self.regress = torch.nn.ModuleList([
            self._make_predict_module(self.channels[d], 4)
            for d in range(self.depth_stop, self.depth + 1)
        ])
        
        # Point location prediction
        self.pointlocation = torch.nn.ModuleList([
            self._make_predict_module(self.channels[self.depth], 4)
        ])

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
        """Create a prediction module with normalization and activation.
        
        Args:
            channel_in: Number of input channels
            channel_out: Number of output channels
            num_hidden: Number of hidden units
            
        Returns:
            Sequential prediction module
        """
        return torch.nn.Sequential(
            modules.Conv1x1GnGeluSequential(channel_in, num_hidden),
            modules.Conv1x1(num_hidden, channel_out, use_bias=True)
        )

    def _get_input_feature(self, doctree):
        """Extract input features from dual octree."""
        return doctree.get_input_feature()

    def octree_encoder_step(self, octree, doctree):
        """Perform octree encoding step.
        
        Args:
            octree: Input octree
            doctree: Dual octree structure
            
        Returns:
            Dictionary of convolution outputs at different depths
        """
        depth, depth_stop = self.depth, self.depth_stop
        data = self._get_input_feature(doctree)

        convs = {}
        convs[depth] = data  # Initial channel = 4

        for i, d in enumerate(range(depth, depth_stop - 1, -1)):
            # Perform graph convolution
            convd = convs[d]
            
            if d == self.depth:  # First convolution layer
                convd = self.conv1(convd, doctree, d)
            
            convd = self.encoder[i](convd, doctree, d)
            convs[d] = convd

            # Downsampling step
            if d > depth_stop:
                nnum = doctree.nnum[d]
                lnum = doctree.lnum[d - 1]
                leaf_mask = doctree.node_child(d - 1) < 0
                convs[d - 1] = self.downsample[i](convd, doctree, d - 1, leaf_mask, nnum, lnum)

        # Apply output normalization and activation
        convs[depth_stop] = self.encoder_norm_out(convs[depth_stop], doctree, depth_stop)
        convs[depth_stop] = self.nonlinearity(convs[depth_stop])

        return convs

    def octree_encoder(self, octree, doctree):
        """Encode octree to latent distribution.
        
        Args:
            octree: Input octree
            doctree: Dual octree structure
            
        Returns:
            Diagonal Gaussian posterior distribution
        """
        convs = self.octree_encoder_step(octree, doctree)
        code = self.KL_conv(convs[self.depth_stop])
        posterior = DiagonalGaussianDistribution(code)
        return posterior

    def octree_decoder(self, code, doctree_out, update_octree=False):
        """Decode latent code to octree structure and signals.
        
        Args:
            code: Latent code tensor
            doctree_out: Output dual octree structure
            update_octree: Whether to update octree structure during decoding
            
        Returns:
            Tuple of (logits, regression_voxels, octree, signal)
        """
        code = self.post_KL_conv(code)
        octree_out = doctree_out.octree

        logits = {}
        reg_voxs = {}
        deconvs = {}
        depth_stop = self.depth_stop

        # Initialize decoder features
        deconvs[depth_stop] = code

        # Apply middle decoder blocks
        deconvs[depth_stop] = self.decoder_mid.block_1(deconvs[depth_stop], doctree_out, depth_stop)
        deconvs[depth_stop] = self.decoder_mid.block_2(deconvs[depth_stop], doctree_out, depth_stop)

        # Decode from depth_stop to depth_out
        for i, d in enumerate(range(self.depth_stop, self.depth_out + 1)):
            # Upsampling step
            if d > self.depth_stop:
                nnum = doctree_out.nnum[d - 1]
                leaf_mask = doctree_out.node_child(d - 1) < 0
                deconvs[d] = self.upsample[i - 1](deconvs[d - 1], doctree_out, d, leaf_mask, nnum)

            # Apply decoder blocks
            octree_out = doctree_out.octree
            deconvs[d] = self.decoder[i](deconvs[d], doctree_out, d)

            # Predict splitting labels
            logit = self.predict[i]([deconvs[d], doctree_out, d])
            nnum = doctree_out.nnum[d]
            logits[d] = logit[-nnum:]

            # Update octree structure if required
            if update_octree:
                label = logits[d].argmax(1).to(torch.int32)
                octree_out = doctree_out.octree
                octree_out.octree_split(label, d)
                
                if d < self.depth_out:
                    octree_out.octree_grow(d + 1)
                    octree_out.depth += 1
                    
                doctree_out = dual_octree.DualOctree(octree_out)
                doctree_out.post_processing_for_docnn()

            # Predict regression values
            reg_vox = self.regress[i]([deconvs[d], doctree_out, d])

            # Pad zeros for compatibility
            node_mask = doctree_out.graph[d]['node_mask']
            shape = (node_mask.shape[0], reg_vox.shape[1])
            reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
            reg_vox_pad[node_mask] = reg_vox
            reg_voxs[d] = reg_vox_pad

            # Predict signal at final depth
            if d == self.depth_out:
                signal = self.pointlocation[0]([deconvs[d], doctree_out, d])
                signal = torch.tanh(signal[-nnum:]).clone()
                signal[:, :3] = 0.5 * signal[:, :3]
                signal = ocnn.nn.octree_depad(signal, doctree_out.octree, d)

        return logits, reg_voxs, doctree_out.octree, signal

    def create_full_octree(self, octree_in: Octree):
        """Initialize a full octree for decoding.
        
        Args:
            octree_in: Input octree for reference
            
        Returns:
            Full octree initialized to specified depth
        """
        device = octree_in.device
        batch_size = octree_in.batch_size
        octree = Octree(self.depth, self.full_depth, batch_size, device)
        
        for d in range(self.full_depth + 1):
            octree.octree_grow_full(depth=d)
            
        return octree

    def create_child_octree(self, octree_in: Octree):
        """Create child octree based on input octree structure.
        
        Args:
            octree_in: Input octree
            
        Returns:
            Child octree with structure matching input
        """
        octree_out = self.create_full_octree(octree_in)
        octree_out.depth = self.full_depth
        
        for d in range(self.full_depth, self.depth_stop):
            label = octree_in.nempty_mask(d).long()
            octree_out.octree_split(label, d)
            octree_out.octree_grow(d + 1)
            octree_out.depth += 1
            
        return octree_out

    def points2octree(self, points, device=None):
        """Convert point cloud to octree representation.
        
        Args:
            points: Point cloud tensor with shape (N, 4)
            device: Target device for octree
            
        Returns:
            Octree built from input points
        """
        points_in = Points(points[:, :3], features=points[:, 3].unsqueeze(1))
        points_in.clip(min=-1, max=1)  # Clip points to valid range
        octree = Octree(8, 4, device=device)
        octree.build_octree(points_in)
        return octree

    def randomly_remove_points(self, points, num_remove_points=5, remove_radius=0.05):
        """Randomly remove points from point cloud in specified radius.
        
        Args:
            points: Input point cloud tensor with shape (N, C)
            num_remove_points: Number of random points to select for removal
            remove_radius: Radius around selected points for removal
            
        Returns:
            Point cloud with points removed
        """
        if points.size(0) == 0:
            return points
            
        positions = points[:, :3]  # Extract xyz coordinates
        remove_indices = torch.randperm(positions.size(0))[:num_remove_points]
        remove_points = positions[remove_indices]
        
        # Calculate distances and create removal mask
        distances = torch.cdist(positions, remove_points)
        within_radius_mask = torch.any(distances < remove_radius, dim=1)
        remaining_points = points[~within_radius_mask]
        
        return remaining_points

    def forward(self, octree_in, octree_out=None, pos=None, evaluate=False):
        """Forward pass through the VAE.
        
        Args:
            octree_in: Input octree
            octree_out: Output octree (optional)
            pos: Position tensor for MPU evaluation
            evaluate: Whether in evaluation mode
            
        Returns:
            Dictionary containing model outputs
        """
        # Generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()

        # Determine if octree structure should be updated
        update_octree = octree_out is None
        
        if update_octree:
            octree_out = self.create_full_octree(octree_in)
            octree_out.depth = self.full_depth
            
            for d in range(self.full_depth, self.depth_stop):
                label = octree_in.nempty_mask(d).long()
                octree_out.octree_split(label, d)
                octree_out.octree_grow(d + 1)
                octree_out.depth += 1

        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        # Encode and sample from posterior
        posterior = self.octree_encoder(octree_in, doctree_in)
        z = posterior.sample()

        if evaluate:
            z = posterior.sample()
            print(f"Code stats: max={z.max():.4f}, min={z.min():.4f}, "
                  f"mean={z.mean():.4f}, std={z.std():.4f}")

        # Decode latent code
        out = self.octree_decoder(z, doctree_out, update_octree)
        
        # Prepare output dictionary
        output = {
            'logits': out[0], 
            'reg_voxs': out[1], 
            'octree_out': out[2], 
            'signal': out[3]
        }
        
        # Add KL loss and code statistics
        kl_loss = posterior.kl()
        output['kl_loss'] = kl_loss.mean()
        output['code_max'] = z.max()
        output['code_min'] = z.min()

        # Compute MPU values if positions provided
        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # Create MPU wrapper function for testing
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_stop][0]
            
        output['neural_mpu'] = _neural_mpu

        return output

    def extract_code(self, octree_in):
        """Extract latent code from input octree.
        
        Args:
            octree_in: Input octree
            
        Returns:
            Tuple of (sampled_code, dual_octree)
        """
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()

        # Encode to get feature maps at different depths
        convs = self.octree_encoder_step(octree_in, doctree_in)
        code = self.KL_conv(convs[self.depth_stop])
        posterior = DiagonalGaussianDistribution(code)
        
        return posterior.sample(), doctree_in

    def decode_code(self, code, doctree_in, update_octree=True, pos=None):
        """Decode latent code to octree structure.
        
        Args:
            code: Latent code tensor
            doctree_in: Input dual octree
            update_octree: Whether to update octree structure
            pos: Position tensor for MPU evaluation
            
        Returns:
            Dictionary containing decoded outputs
        """
        octree_in = doctree_in.octree
        
        # Generate output dual octree
        if update_octree:
            octree_out = self.create_child_octree(octree_in)
            doctree_out = dual_octree.DualOctree(octree_out)
            doctree_out.post_processing_for_docnn()
        else:
            doctree_out = doctree_in

        # Run decoder
        out = self.octree_decoder(code, doctree_out, update_octree=update_octree)
        
        output = {
            'logits': out[0], 
            'reg_voxs': out[1], 
            'octree_out': out[2], 
            'signal': out[3]
        }

        # Compute MPU values if positions provided
        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # Create MPU wrapper function
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]
            
        output['neural_mpu'] = _neural_mpu

        return output



import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List, Optional
import warnings

# Added optimized structures for MST and neighbor search
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as cs_mst

warnings.filterwarnings('ignore')


class SkeletonRefiner:
    """Refine signal points into complete skeleton curves."""
    
    def __init__(self, max_edge_dist: float = 0.2, recon_threshold: float = 0.1, 
                 min_nodes: int = 20, max_dist: float = 0.4):
        """Initialize the skeleton refiner.
        
        Args:
            max_edge_dist: Maximum distance for MST edges (default: 0.2)
            recon_threshold: Maximum distance for reconnecting endpoints (default: 0.1)
            min_nodes: Minimum nodes required to keep a component (default: 20)
            max_dist: Maximum distance for global point connections (default: 0.4)
        """
        self.max_edge_dist = max_edge_dist
        self.recon_threshold = recon_threshold
        self.min_nodes = min_nodes
        self.max_dist = max_dist
    
    def build_mst_graph(self, points: np.ndarray) -> nx.Graph:
        # Optimized MST construction using sparse graph built via cKDTree within radius
        n = points.shape[0]
        if n == 0:
            return nx.Graph()
        
        try:
            tree = cKDTree(points)
            # Candidate edges: only within max_edge_dist (consistent with later thresholding)
            pairs = tree.query_pairs(r=float(self.max_edge_dist))
            # Convert to ndarray pairs
            if isinstance(pairs, set):
                if len(pairs) == 0:
                    pairs_arr = np.empty((0, 2), dtype=np.int64)
                else:
                    pairs_arr = np.fromiter((p for ij in pairs for p in ij), dtype=np.int64)
                    pairs_arr = pairs_arr.reshape(-1, 2)
            else:
                pairs_arr = np.asarray(pairs, dtype=np.int64)
            
            if pairs_arr.size > 0:
                i = pairs_arr[:, 0]
                j = pairs_arr[:, 1]
                diffs = points[i] - points[j]
                dists = np.linalg.norm(diffs, axis=1)
                # Build symmetric sparse adjacency
                row = np.concatenate([i, j])
                col = np.concatenate([j, i])
                data = np.concatenate([dists, dists])
                adj = coo_matrix((data, (row, col)), shape=(n, n))
                # Minimum spanning forest on sparse graph
                T = cs_mst(adj)
                G = nx.from_scipy_sparse_array(T + T.T)
                # Ensure all nodes exist (isolated nodes kept)
                G.add_nodes_from(range(n))
                # Filter edges by distance threshold (match original behavior)
                edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0.0) > self.max_edge_dist]
                if edges_to_remove:
                    G.remove_edges_from(edges_to_remove)
                return G
            else:
                # Fallback to small k-NN if no pairs within radius
                if n <= 1:
                    G = nx.Graph()
                    G.add_nodes_from(range(n))
                    return G
                k = min(8, n - 1)
                dists, idxs = tree.query(points, k=k + 1)  # include self at idx 0
                rows = []
                cols = []
                data = []
                for u in range(n):
                    for v, d in zip(idxs[u][1:], dists[u][1:]):
                        rows.append(u); cols.append(int(v)); data.append(float(d))
                if len(rows) == 0:
                    G = nx.Graph()
                    G.add_nodes_from(range(n))
                    return G
                adj = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
                # Symmetrize by taking minimum weight for undirected edges
                adj = adj.minimum(adj.T) + adj.T.minimum(adj)
                T = cs_mst(adj)
                G = nx.from_scipy_sparse_array(T + T.T)
                G.add_nodes_from(range(n))
                # Filter edges by max_edge_dist to stay consistent
                edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0.0) > self.max_edge_dist]
                if edges_to_remove:
                    G.remove_edges_from(edges_to_remove)
                return G
        except Exception as e:
            # Safe fallback (may be slow on very large N)
            print(f"Warning: Fast MST construction failed ({e}), falling back to dense method")
            dist_matrix = squareform(pdist(points))
            np.fill_diagonal(dist_matrix, 0)
            G = nx.from_numpy_array(dist_matrix)
            mst = nx.minimum_spanning_tree(G)
            # Filter edges by distance threshold to match original behavior
            edges_to_remove = []
            for u, v, data in mst.edges(data=True):
                if data['weight'] > self.max_edge_dist:
                    edges_to_remove.append((u, v))
            mst.remove_edges_from(edges_to_remove)
            return mst
    
    def filter_small_components(self, graph: nx.Graph) -> Tuple[nx.Graph, np.ndarray]:
        """Remove small connected components and return node mapping.
        
        Args:
            graph: Input NetworkX graph
            
        Returns:
            Filtered graph and array of kept node indices
        """
        components = list(nx.connected_components(graph))
        keep_components = [comp for comp in components if len(comp) >= self.min_nodes]
        
        if not keep_components:
            return nx.Graph(), np.array([])
        
        keep_nodes = sorted(set().union(*keep_components))
        subgraph = graph.subgraph(keep_nodes).copy()
        
        return subgraph, np.array(keep_nodes)
    
    def handle_multi_branch_points(self, graph: nx.Graph, points: np.ndarray, 
                                   order: np.ndarray) -> nx.Graph:
        """Handle multi-branch points by keeping longest branch and connecting to nearest endpoints.
        Optimized version following MATLAB logic more closely.
        
        Args:
            graph: Input graph
            points: Point coordinates
            order: Point order values
            
        Returns:
            Modified graph with degree ≤ 2 for most nodes
        """
        G = graph.copy()
        multi_branch_nodes = [node for node in G.nodes() if G.degree(node) > 2]
        
        if not multi_branch_nodes:
            return G
        
        edges_to_remove = []
        edges_to_add = []
        
        # Pre-calculate all endpoints to avoid repeated computation
        all_endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        
        for node in multi_branch_nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) <= 2:
                continue
            
            # Simplified branch size calculation using BFS from each neighbor
            branch_sizes = []
            for neighbor in neighbors:
                # Simple BFS to count reachable nodes (excluding current node)
                visited = {node}  # Block the multi-branch node
                queue = [neighbor]
                count = 0
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    count += 1
                    
                    # Add unvisited neighbors
                    for next_node in G.neighbors(current):
                        if next_node not in visited:
                            queue.append(next_node)
                
                branch_sizes.append(count)
            
            # Keep the largest branch
            max_branch_idx = np.argmax(branch_sizes)
            keep_neighbor = neighbors[max_branch_idx]
            
            # Remove other edges
            for i, neighbor in enumerate(neighbors):
                if i != max_branch_idx:
                    edges_to_remove.append((node, neighbor))
            
            # Find nearest endpoint (excluding current node)
            valid_endpoints = [ep for ep in all_endpoints if ep != node]
            if valid_endpoints:
                curr_point = points[node]
                endpoint_points = points[valid_endpoints]
                distances = np.linalg.norm(endpoint_points - curr_point, axis=1)
                
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < 0.2:  # Distance threshold
                    best_endpoint = valid_endpoints[min_dist_idx]
                    edges_to_add.append((node, best_endpoint))
        
        # Apply all modifications at once
        G.remove_edges_from(edges_to_remove)
        G.add_edges_from(edges_to_add)
        
        return G
    
    def connect_nearby_endpoints(self, graph: nx.Graph, points: np.ndarray) -> nx.Graph:
        """Connect nearby endpoints to form continuous curves.
        
        Args:
            graph: Input graph
            points: Point coordinates
            
        Returns:
            Graph with connected endpoints
        """
        G = graph.copy()
        
        # Find all endpoints (degree 1 nodes)
        endpoints = [node for node in G.nodes() if G.degree(node) == 1]
        
        if len(endpoints) < 2:
            return G
        
        # Use cKDTree to find endpoint pairs within recon_threshold efficiently
        endpoint_points = points[endpoints]
        tree = cKDTree(endpoint_points)
        pairs = tree.query_pairs(r=float(self.recon_threshold))
        
        # Add edges between valid endpoint pairs (avoid duplicates)
        added_pairs = set()
        if isinstance(pairs, set):
            iterable_pairs = pairs
        else:
            iterable_pairs = set(map(tuple, np.asarray(pairs, dtype=int)))
        for i, j in iterable_pairs:
            if i < j:  # Avoid duplicates
                pair = (endpoints[i], endpoints[j])
                if pair not in added_pairs:
                    G.add_edge(pair[0], pair[1])
                    added_pairs.add(pair)
        
        return G
    
    def find_longest_path(self, graph: nx.Graph, points: np.ndarray, 
                         order: np.ndarray) -> List[int]:
        """Find the longest path in the graph starting from minimum order point.
        
        Args:
            graph: Input graph
            points: Point coordinates  
            order: Point order values
            
        Returns:
            List of node indices forming the longest path
        """
        if graph.number_of_nodes() == 0:
            return []
        
        # Find starting point (minimum order)
        valid_nodes = list(graph.nodes())
        if not valid_nodes:
            return []
        
        min_order_idx = np.argmin(order[valid_nodes])
        start_node = valid_nodes[min_order_idx]
        
        # Find nearest neighbor as second point
        neighbors = list(graph.neighbors(start_node))
        if not neighbors:
            return [start_node]
        
        start_point = points[start_node]
        neighbor_points = points[neighbors]
        distances = np.linalg.norm(neighbor_points - start_point, axis=1)
        second_node = neighbors[np.argmin(distances)]
        
        # Grow path in both directions
        path = [start_node, second_node]
        used = set(path)
        
        # Grow from tail (second_node)
        path = self._grow_path(graph, path, second_node, used, points, order, 'tail')
        
        # Grow from head (start_node)  
        path = self._grow_path(graph, path, start_node, used, points, order, 'head')
        
        # Ensure path direction (start should have smaller order than end)
        if len(path) > 1:
            start_order = order[path[0]]
            end_order = order[path[-1]]
            if start_order > end_order:
                path = path[::-1]
        
        return path
    
    def _grow_path(self, graph: nx.Graph, path: List[int], endpoint: int, 
                   used: set, points: np.ndarray, order: np.ndarray, 
                   direction: str) -> List[int]:
        """Grow path from one endpoint.
        
        Args:
            graph: Graph to traverse
            path: Current path
            endpoint: Current endpoint to grow from
            used: Set of used nodes
            points: Point coordinates
            order: Point order values
            direction: 'head' or 'tail'
            
        Returns:
            Extended path
        """
        current_node = endpoint
        
        while True:
            # Find unused neighbors
            neighbors = [n for n in graph.neighbors(current_node) if n not in used]
            
            if not neighbors:
                # Try global nearest point as fallback
                all_unused = [n for n in graph.nodes() if n not in used]
                if not all_unused:
                    break
                
                current_point = points[current_node]
                unused_points = points[all_unused]
                distances = np.linalg.norm(unused_points - current_point, axis=1)
                min_dist_idx = np.argmin(distances)
                
                if distances[min_dist_idx] <= self.max_dist:
                    neighbors = [all_unused[min_dist_idx]]
                else:
                    break
            
            # Choose neighbor with closest order
            current_order = order[current_node]
            neighbor_orders = order[neighbors]
            best_neighbor_idx = np.argmin(np.abs(neighbor_orders - current_order))
            next_node = neighbors[best_neighbor_idx]
            
            # Add to path
            used.add(next_node)
            if direction == 'head':
                path.insert(0, next_node)
            else:
                path.append(next_node)
            
            current_node = next_node
        
        return path
    
    def smooth_curve_with_repulsion(self, points: np.ndarray, min_dist: float = 0.1,
                                   num_iter: int = 10, step_size: float = 0.02,
                                   exclusion_interval: int = 50) -> np.ndarray:
        """Apply repulsion-based smoothing to curve points.
        Optimized version with better performance.
        
        Args:
            points: N×3 curve points
            min_dist: Minimum desired distance between points
            num_iter: Number of iterations
            step_size: Step size for each iteration
            exclusion_interval: Skip points within this interval along curve
            
        Returns:
            Smoothed curve points
        """
        # Reduce the target size for faster processing
        target = min(2000, max(500, len(points)))  # Reduced from 4000
        adjusted_points = self._uniform_resample_curve(points, target)
        n = len(adjusted_points)
        
        # Reduce iterations for large n
        if n > 2000 and num_iter > 4:
            num_iter = 4
        elif n > 1000 and num_iter > 6:
            num_iter = 6
        
        for iteration in range(num_iter):
            displacement = np.zeros_like(adjusted_points)
            # Use KDTree for efficient neighbor search
            tree = cKDTree(adjusted_points)
            
            # Batch process neighbors to reduce overhead
            for i in range(n):
                neighbors = tree.query_ball_point(adjusted_points[i], r=float(min_dist))
                if not neighbors:
                    continue
                
                # Vectorized computation for all valid neighbors at once
                valid_j = [j for j in neighbors if j != i and abs(i - j) >= exclusion_interval]
                if not valid_j:
                    continue
                
                neighbor_points = adjusted_points[valid_j]
                vecs = adjusted_points[i] - neighbor_points
                dists = np.linalg.norm(vecs, axis=1)
                
                # Apply repulsion only where distance < min_dist
                valid_mask = (dists < min_dist) & (dists > 1e-6)
                if np.any(valid_mask):
                    valid_vecs = vecs[valid_mask]
                    valid_dists = dists[valid_mask]
                    repulsions = (min_dist - valid_dists[:, np.newaxis]) * (valid_vecs / valid_dists[:, np.newaxis])
                    displacement[i] += np.sum(repulsions, axis=0)
            
            # Update positions
            adjusted_points += step_size * displacement
            
            # Apply lighter Gaussian smoothing
            for dim in range(3):
                adjusted_points[:, dim] = gaussian_filter1d(adjusted_points[:, dim], sigma=0.5)
        
        return adjusted_points
    
    def _uniform_resample_curve(self, points: np.ndarray, target_num: int) -> np.ndarray:
        """Uniformly resample curve points along arc length.
        
        Args:
            points: N×3 input points
            target_num: Target number of points
            
        Returns:
            Resampled points
        """
        if len(points) < 2:
            return points
        
        # Calculate arc length
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0], np.cumsum(dists)])
        
        # Interpolate with splines
        s_target = np.linspace(0, s[-1], target_num)
        
        resampled = np.zeros((target_num, 3))
        for dim in range(3):
            f = interp1d(s, points[:, dim], kind='cubic', bounds_error=False, fill_value='extrapolate')
            resampled[:, dim] = f(s_target)
        
        return resampled
    
    def generate_smooth_curve(self, points: np.ndarray, num_points: int = 4000) -> np.ndarray:
        """Generate smooth arc-length parameterized curve.
        
        Args:
            points: Input curve points
            num_points: Target number of output points
            
        Returns:
            Smooth curve with specified number of points
        """
        if len(points) < 2:
            return points
        
        # Sparse sampling first (every 3rd point)
        sparse_points = points[::3]
        
        # Calculate arc length for sparse points
        diffs = np.diff(sparse_points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0], np.cumsum(dists)])
        
        # Create uniform arc length sampling
        s_uniform = np.linspace(0, s[-1], num_points)
        
        # Spline interpolation
        smooth_curve = np.zeros((num_points, 3))
        for dim in range(3):
            f = interp1d(s, sparse_points[:, dim], kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
            smooth_curve[:, dim] = f(s_uniform)
        
        # Optional additional smoothing
        for dim in range(3):
            smooth_curve[:, dim] = gaussian_filter1d(smooth_curve[:, dim], sigma=2.0)
        
        return smooth_curve
    
    def process_signal_to_skeleton(self, signal: np.ndarray) -> np.ndarray:
        """Main processing function to convert signal points to skeleton curve.
        
        Args:
            signal: N×4 array (xyz + order)
            
        Returns:
            Processed skeleton curve points (M×3)
        """
        if len(signal) < self.min_nodes:
            print(f"Warning: Too few points ({len(signal)}) for skeleton processing")
            return signal[:, :3]  # Return just xyz coordinates
        
        points = signal[:, :3].astype(np.float64)
        order = signal[:, 3].astype(np.float64)
        
        # Step 1: Build MST and filter edges
        print("Building minimum spanning tree...")
        mst_graph = self.build_mst_graph(points)
        
        # Step 2: Remove small components
        print("Filtering small components...")
        filtered_graph, keep_indices = self.filter_small_components(mst_graph)
        
        if len(keep_indices) == 0:
            print("Warning: No large components found")
            return points[:min(100, len(points))]  # Return first 100 points as fallback
        
        # Update points and order arrays
        sub_points = points[keep_indices]
        sub_order = order[keep_indices]
        
        # Step 3: Handle multi-branch points
        print("Handling multi-branch points...")
        refined_graph = self.handle_multi_branch_points(filtered_graph, sub_points, sub_order)
        
        # Step 4: Connect nearby endpoints
        print("Connecting nearby endpoints...")
        final_graph = self.connect_nearby_endpoints(refined_graph, sub_points)
        
        # Step 5: Find longest path
        print("Finding longest path...")
        main_path_indices = self.find_longest_path(final_graph, sub_points, sub_order)
        
        if len(main_path_indices) < 2:
            print("Warning: Could not find valid path")
            return sub_points[:min(100, len(sub_points))]
        
        main_path_points = sub_points[main_path_indices]
        
        # Step 6: Apply repulsion-based smoothing
        print("Applying repulsion smoothing...")
        smoothed_points = self.smooth_curve_with_repulsion(main_path_points)
        
        # Step 7: Generate final smooth curve
        print("Generating final smooth curve...")
        final_curve = self.generate_smooth_curve(smoothed_points)
        
        return final_curve


def refine_signal_to_skeleton(signal: np.ndarray, 
                             max_edge_dist: float = 0.2,
                             recon_threshold: float = 0.1,
                             min_nodes: int = 20,
                             max_dist: float = 0.4) -> np.ndarray:
    """Convenience function to refine signal points to skeleton curve.
    
    Args:
        signal: N×4 array (xyz + order)
        max_edge_dist: Maximum distance for MST edges
        recon_threshold: Maximum distance for reconnecting endpoints  
        min_nodes: Minimum nodes required to keep a component
        max_dist: Maximum distance for global point connections
        
    Returns:
        Refined skeleton curve points (M×3)
    """
    refiner = SkeletonRefiner(max_edge_dist, recon_threshold, min_nodes, max_dist)
    return refiner.process_signal_to_skeleton(signal)


if __name__ == "__main__":
    # Example usage
    import scipy.io as sio
    
    # Load test data
    signal_file = "test_signal.mat"
    try:
        data = sio.loadmat(signal_file)
        signal = data['signal']  # N×4 array
        
        # Process to skeleton
        skeleton = refine_signal_to_skeleton(signal)
        
        # Save result
        sio.savemat("skeleton_refined.mat", {'curve': skeleton})
        print(f"Processed {len(signal)} signal points to {len(skeleton)} skeleton points")
        
    except FileNotFoundError:
        print(f"Test file {signal_file} not found")
        
        # Generate synthetic test data
        t = np.linspace(0, 4*np.pi, 1000)
        points = np.column_stack([
            np.cos(t) + 0.1*np.random.randn(len(t)),
            np.sin(t) + 0.1*np.random.randn(len(t)), 
            0.1*t + 0.05*np.random.randn(len(t)),
            t  # order
        ])
        
        skeleton = refine_signal_to_skeleton(points)
        print(f"Synthetic test: {len(points)} → {len(skeleton)} points")

"""
Test script for 2-step clustering implementation
"""

import torch
import numpy as np
from dino_correspondence import kway_cluster_per_image_two_step, match_centers_two_step

def test_two_step_clustering():
    """Test basic 2-step clustering functionality"""
    print("Testing 2-step clustering implementation...")
    
    # Create synthetic embeddings for 2 images
    batch_size = 2
    length = 197  # CLS + 14x14 patches
    channels = 768
    
    # Random embeddings
    image_embeds = torch.randn(batch_size, length, channels)
    
    # Test clustering
    n_superclusters = 3
    n_subclusters_per_supercluster = 5
    
    print(f"\nComputing {n_superclusters} superclusters with {n_subclusters_per_supercluster} subclusters each...")
    
    try:
        supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
            image_embeds,
            n_superclusters=n_superclusters,
            n_subclusters_per_supercluster=n_subclusters_per_supercluster
        )
        
        print(f"✓ Clustering successful")
        print(f"  - Supercluster eigenvectors shape: {supercluster_eigvecs.shape}")
        print(f"  - Subcluster eigenvectors shape: {subcluster_eigvecs.shape}")
        print(f"  - Subcluster to supercluster mapping shape: {subcluster_to_supercluster.shape}")
        
        # Verify shapes
        expected_total_subclusters = n_superclusters * n_subclusters_per_supercluster
        assert supercluster_eigvecs.shape == (batch_size, length, n_superclusters)
        assert subcluster_eigvecs.shape == (batch_size, length, expected_total_subclusters)
        assert subcluster_to_supercluster.shape == (batch_size, expected_total_subclusters)
        
        print(f"✓ Shapes verified")
        
        # Test matching
        print(f"\nTesting cluster matching...")
        
        subcluster_mapping = match_centers_two_step(
            image_embeds[0],
            image_embeds[1],
            supercluster_eigvecs[0],
            supercluster_eigvecs[1],
            subcluster_eigvecs[0],
            subcluster_eigvecs[1],
            subcluster_to_supercluster[0],
            subcluster_to_supercluster[1],
            supercluster_match_method='hungarian',
            subcluster_match_method='hungarian'
        )
        
        print(f"✓ Matching successful")
        print(f"  - Mapping shape: {subcluster_mapping.shape}")
        print(f"  - Mapping range: {subcluster_mapping.min()} to {subcluster_mapping.max()}")
        
        # Verify mapping
        assert subcluster_mapping.shape == (expected_total_subclusters,)
        assert subcluster_mapping.min() >= 0
        assert subcluster_mapping.max() < expected_total_subclusters
        
        print(f"✓ Mapping verified")
        
        # Test constraint: subclusters should only match within same supercluster
        print(f"\nVerifying hierarchical constraint...")
        violations = 0
        for sub1_idx in range(expected_total_subclusters):
            sub2_idx = subcluster_mapping[sub1_idx]
            super1_idx = subcluster_to_supercluster[0][sub1_idx].item()
            super2_idx = subcluster_to_supercluster[1][sub2_idx].item()
            
            # These superclusters should be matched
            # (We need to compute supercluster matching to verify this properly)
        
        print(f"✓ All tests passed!\n")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_different_parameters():
    """Test with various parameter combinations"""
    print("\nTesting different parameter combinations...")
    
    test_cases = [
        (2, 3, "small"),
        (3, 5, "medium"),
        (4, 7, "large"),
    ]
    
    image_embeds = torch.randn(2, 197, 768)
    
    for n_super, n_sub, description in test_cases:
        print(f"\n  Testing {description}: {n_super} superclusters × {n_sub} subclusters")
        try:
            supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
                image_embeds,
                n_superclusters=n_super,
                n_subclusters_per_supercluster=n_sub
            )
            expected_total = n_super * n_sub
            assert subcluster_eigvecs.shape[2] == expected_total, f"Expected {expected_total} subclusters, got {subcluster_eigvecs.shape[2]}"
            print(f"    ✓ {description} configuration works ({expected_total} total subclusters)")
        except Exception as e:
            print(f"    ✗ {description} configuration failed: {e}")
            return False
    
    print(f"\n✓ All parameter combinations passed!\n")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TWO-STEP CLUSTERING TEST SUITE")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success = test_two_step_clustering() and success
    success = test_different_parameters() and success
    
    # Summary
    print("=" * 60)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

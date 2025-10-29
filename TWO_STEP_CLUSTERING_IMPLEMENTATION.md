# Two-Step Clustering Implementation

## Overview

This implementation adds a refined 2-step hierarchical clustering technique to the image interpolation system. The approach first identifies coarse "superclusters" across images, matches them, then computes finer "subclusters" within each matched supercluster pair.

## Key Concepts

### Traditional Single-Step Clustering
- Computes N clusters directly across images
- Matches all clusters globally between images

### New Two-Step Clustering  
1. **Step 1: Supercluster Discovery**
   - Compute a small number of coarse superclusters (e.g., 3) across all input images
   - Match superclusters between images using specified matching method (hungarian/argmin)

2. **Step 2: Subcluster Refinement**
   - Within each supercluster, compute finer subclusters (e.g., 5 per supercluster)
   - Match subclusters ONLY within corresponding matched supercluster pairs
   - This ensures subclusters can only match to other subclusters in the same semantic region

### Benefits
- **Better semantic coherence**: Subclusters are constrained to match within the same supercluster
- **Hierarchical structure**: Coarse-to-fine matching mirrors natural visual organization
- **Controlled granularity**: Separate control over coarse and fine clustering levels
- **Independent matching methods**: Can use different strategies for superclusters vs subclusters

## Implementation Details

### New Functions in `dino_correspondence.py`

#### `kway_cluster_per_image_two_step()`
Performs 2-step hierarchical clustering on image embeddings.

**Parameters:**
- `image_embeds`: (batch, length, channels) - Image embeddings
- `n_superclusters`: Number of coarse superclusters (e.g., 3)
- `n_subclusters_per_supercluster`: Number of subclusters per supercluster (e.g., 5)
- `supercluster_gamma`, `subcluster_gamma`: NCut parameters (None = auto)
- `degree`: Degree parameter for gamma estimation

**Returns:**
- `supercluster_eigenvectors`: (batch, length, n_superclusters)
- `subcluster_eigenvectors`: (batch, length, total_subclusters)
- `subcluster_to_supercluster_mapping`: (batch, total_subclusters) - maps each subcluster to its parent supercluster

**Total clusters** = n_superclusters × n_subclusters_per_supercluster

#### `match_centers_two_step()`
Matches clusters using the 2-step hierarchical approach.

**Process:**
1. Match superclusters between images
2. For each matched supercluster pair:
   - Extract subclusters belonging to these superclusters
   - Match subclusters within this constrained set
3. Return global subcluster mapping

**Parameters:**
- Supercluster and subcluster eigenvectors for both images
- Mappings from subclusters to superclusters
- Separate matching methods for each level

### Updated Functions in `app.py`

#### `perform_two_image_interpolation()`
New parameters:
- `use_two_step_clustering`: Enable 2-step mode (default: False)
- `n_superclusters`: Number of superclusters (default: 3)
- `n_subclusters_per_supercluster`: Subclusters per supercluster (default: 5)
- `supercluster_match_method`: Matching for superclusters (default: 'hungarian')
- `subcluster_match_method`: Matching for subclusters (default: 'hungarian')

#### `perform_n_image_interpolation()`
Same new parameters as above for N-image interpolation.

#### `perform_n_image_interpolation_per_cluster()`
Enhanced to support 2-step clustering with per-cluster weight control.

### UI Changes

#### Tab 2: Two-Image Interpolation
Added clustering method selector:
- **Checkbox**: "Use 2-Step Hierarchical Clustering"
- **Single-step controls** (visible when unchecked):
  - Number of Clusters slider
  - Matching Method radio buttons
- **Two-step controls** (visible when checked):
  - Number of Superclusters slider (2-10)
  - Subclusters per Supercluster slider (2-10)
  - Supercluster Matching Method
  - Subcluster Matching Method

#### Tab 4: N-Image Interpolation
Similar controls added for N-image cluster-aware interpolation with dynamic visibility toggling.

## Usage Examples

### Two-Image Interpolation with 2-Step Clustering

```python
# Traditional single-step (15 clusters total)
result_images = perform_two_image_interpolation(
    image1, image2, model, weights,
    n_clusters=15,
    match_method='hungarian'
)

# Two-step hierarchical (3 superclusters × 5 subclusters = 15 clusters total)
result_images = perform_two_image_interpolation(
    image1, image2, model, weights,
    use_two_step_clustering=True,
    n_superclusters=3,
    n_subclusters_per_supercluster=5,
    supercluster_match_method='hungarian',
    subcluster_match_method='hungarian'
)
```

### N-Image Interpolation with 2-Step Clustering

```python
# With 2-step clustering
result_images = perform_n_image_interpolation(
    image_list,
    base_image_idx=0,
    model=model,
    interpolation_weights=weights_list,
    use_two_step_clustering=True,
    n_superclusters=3,
    n_subclusters_per_supercluster=5,
    supercluster_match_method='hungarian',
    subcluster_match_method='argmin'
)
```

## Recommended Settings

### For Coarse Blending
- **Superclusters**: 2-3
- **Subclusters per supercluster**: 3-5
- **Total clusters**: 6-15

### For Fine Blending  
- **Superclusters**: 3-4
- **Subclusters per supercluster**: 5-7
- **Total clusters**: 15-28

### Matching Methods
- **Hungarian**: More globally optimal but slower
- **Argmin**: Faster but may produce suboptimal matches
- **Recommended**: Hungarian for superclusters, either method for subclusters

## Technical Notes

### Cluster Indexing
- Subclusters are indexed globally: 0 to (n_superclusters × n_subclusters_per_supercluster - 1)
- Each subcluster tracks its parent supercluster via `subcluster_to_supercluster_mapping`

### Empty Clusters
- If a supercluster has too few tokens for subclustering, dummy subclusters are created
- Empty subclusters have zero eigenvector values

### Precomputed Values
The N-image interpolation UI computes clusters once and reuses them for generation, ensuring consistency between visualization and generation.

## Backward Compatibility

All existing code continues to work without modification:
- Default `use_two_step_clustering=False` maintains original behavior
- All new parameters have sensible defaults
- Single-step clustering remains the default in the UI

## Future Enhancements

Potential improvements:
1. **3+ step clustering**: Extend to deeper hierarchies
2. **Adaptive subclustering**: Different subcluster counts per supercluster
3. **Cross-supercluster blending**: Allow limited matching across supercluster boundaries
4. **Visualization**: Show supercluster boundaries in the UI
5. **Auto-tuning**: Automatically determine optimal supercluster/subcluster counts

## Files Modified

1. `dino_correspondence.py`:
   - Added `kway_cluster_per_image_two_step()`
   - Added `match_centers_two_step()`

2. `app.py`:
   - Added `compute_direction_from_two_images_two_step()`
   - Updated `perform_two_image_interpolation()`
   - Updated `perform_n_image_interpolation()`
   - Updated `perform_n_image_interpolation_per_cluster()`
   - Updated `compute_cluster_interface()`
   - Updated `generate_cluster_image()`
   - Added UI controls in Tabs 2 and 4

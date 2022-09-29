# What is a NeRF?

- One thing you may find surprising about NeRFs: although they learn directly from image data, they use neither convolutional nor transformer layers (at least not the original). An understated benefit of NeRFs is compression; at 5–10MB, the weights of a NeRF model may be smaller than the collection of images used to train them.

- NeRFs by contrast rely on an old yet elegant concept called light fields, or radiance fields. A light field is a function that describes how light transport occurs throughout a 3D volume. It describes the direction of light rays moving through every x=(x, y, z) coordinate in space and in every direction d, described either as θ and ϕ angles or a unit vector. Collectively they form a 5D feature space that describes light transport in a 3D scene. The NeRF, inspired by this representation, attempts to approximate a function that maps from this space into a 4D space consisting of color c=(R,G,B) and a density σ, which you can think of as the likelihood that the light ray at this 5D coordinate space is terminated (e.g. by occlusion). The standard NeRF is thus a function of the form F : (x,d) -> (c,σ).

## NeRF Architecture

- Positional encoding

- The radiance field function approximator (in this case, an MLP)

- Differentiable volume renderer

- Stratified sampling

- Hierarchical volume sampling


### Positional Encoding

It maps its continuous input to a higher-dimensional space using high-frequency functions to aid the model in learning high frequency variations in the data, which leads to sharper models.

```python
class PositionalEncoder(nn.Module):
  r"""
  Sine-cosine positional encoder for input points.
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Alternate sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    r"""
    Apply positional encoding to input.
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
```

### Radiance Field Function

Radiance field function was represented by the NeRF model, a fairly typical multilayer perceptron that takes encoded 3D points and view directions as inputs and returns RGBA values as outputs. While this paper uses a neural network, any function approximator can be used here. 

The NeRF model is 8 layers deep with feature dimension of 256 for most layers. A residual connection is placed at layer 4. After these layers, the RGB and σ values are produced. The RGB values are further processed with a linear layer, then concatenated with the view directions, then passed through yet another linear layer before finally being recombined with σ at the output.

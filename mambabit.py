import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from tqdm.auto import tqdm
import sys
dim_model = 4096
n_vocab = 2
n_layers = 4


@torch.no_grad()
def string_to_bits(text: str, _cache = []) -> Tensor:
   all_values = torch.arange(0, 256)
   if not _cache:
      bits = [((all_values & (1 << i)) != 0).int() for i in range(7, -1, -1)]
      bits_tensor = torch.stack(bits).mT
      _cache.append(bits_tensor)
   else:
      bits_tensor = _cache[0]
   binary = text.encode()
   raw =  torch.frombuffer(binary, dtype=torch.uint8).int()   
   return bits_tensor[raw].long().ravel()

@torch.no_grad()
def bits_to_string(bits: Tensor):
    if bits.dim() == 2:
        return [bits_to_string(t) for t in bits]
    assert bits.dim() == 1
    assert len(bits) % 8 == 0
    factors = torch.tensor([2**i for i in range(7,-1,-1)]).to(device=bits.device)
    as_bytes = bits.view(-1, 8)
    as_bytes = (as_bytes*factors).sum(-1)
    return ''.join([chr(x) for x in as_bytes])

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim_model)
        self.emb.weight.data *= 0.001

    def forward(self, x):
        return self.emb(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(dim_model)
        self.decoder = nn.Linear(dim_model, n_vocab, False)
        self.decoder.weight.data *= 0.001

    def forward(self, x):
        x = self.norm(x)
        x = self.decoder(x)
        return x

class MambaBit(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.layers = nn.ModuleList([Mamba(dim_model) for _ in range(n_layers)])
        self.dec = Decoder()

    def forward(self, x):        
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x

class MambaBitWithInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.layers = nn.ModuleList([Mamba(dim_model, layer_idx=i) for i in range(n_layers)])
        self.dec = Decoder()

    def forward(self, x, inference_parms=None):
        x = self.enc(x)        
        for i,layer in enumerate(self.layers):
            x = layer(x, inference_params=inference_parms)
        x = self.dec(x)
        return x

# test using O(N^2) cacheless stateless algorithm.
@torch.no_grad()
def test_n2(m: MambaBit, prompt: str, chars=10):
    x = string_to_bits(prompt).cuda()[None]
    process = chars * 8
    for i in tqdm(range(process)):
        y = m(x)
        new = y[:, -1:].argmax(-1)
        x = torch.cat((x, new), 1)
    return bits_to_string(x)

# test using O(N) by reusing state
@torch.no_grad()
def test_n(m: MambaBit, prompt: str, chars=10):
    x = string_to_bits(prompt).cuda()[None]
    process = chars * 8

    inference_parms = InferenceParams(
        max_seqlen=x.numel() + process, 
        max_batch_size=1)

    y = m(x, inference_parms=inference_parms)
    new = y[:, -1:].argmax(-1)
    for i in tqdm(range(process)):        
        x = torch.cat((x, new), 1)
        inference_parms.seqlen_offset = x.numel() + i
        y = m(new, inference_parms=inference_parms)
        new = y[:, -1:].argmax(-1)
    return bits_to_string(x)

def run():
    mamba_bit = MambaBitWithInference().cuda()
    mamba_bit.load_state_dict(torch.load("mamba_bit.bin"))


    prompt="FIRST CITIZE" if len(sys.argv) != 2 else sys.argv[1]
    # test_n2 is O(N^2), test_n is O(N) but inference_params are not well documented
    s = test_n(mamba_bit, prompt, chars=256)[0]
    print(s)

if __name__ == "__main__":
    run()
import torch


class Bf16AddChain16(torch.nn.Module):
    def forward(
        self,
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
    ):
        value = torch.add(x0, x1)
        value = torch.add(value, x2)
        value = torch.add(value, x3)
        value = torch.add(value, x4)
        value = torch.add(value, x5)
        value = torch.add(value, x6)
        value = torch.add(value, x7)
        value = torch.add(value, x8)
        value = torch.add(value, x9)
        value = torch.add(value, x10)
        value = torch.add(value, x11)
        value = torch.add(value, x12)
        value = torch.add(value, x13)
        value = torch.add(value, x14)
        value = torch.add(value, x15)
        return torch.add(value, x16)


def create_case():
    torch.manual_seed(20260724)
    inputs = tuple((torch.randn(8, 16) * 0.125).to(torch.bfloat16) for _ in range(17))
    return {
        "name": "bf16_add_chain16",
        "model": Bf16AddChain16().eval(),
        "inputs": inputs,
    }

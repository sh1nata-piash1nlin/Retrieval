import torch
checkpoint = torch.load("/workspace/huy_aichallenge/models/InternVideo/InternVideo2/multi_modality/weights/InternVideo2-stage2_1b-224p-f4.pt", map_location="cpu")
print(checkpoint.keys())
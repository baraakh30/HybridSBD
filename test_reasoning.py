"""Test the reasoning module"""
from detection.hybrid_boundary_detector import create_hybrid_detector
import torch

print('\n' + '='*80)
print('TESTING INTERPRETABLE REASONING MODULE')
print('='*80)

model = create_hybrid_detector(main_dim=384, num_fusion_blocks=4)
model.eval()

batch = torch.randn(1, 128, 3, 48, 64)
boundary, types, reasoning = model(batch)

print('\n✅ Model thinks step-by-step!')
print(f'\nReasoning outputs (what the model "thinks"):')
print(f'  1. Change detection: {reasoning["change_score"].shape}')
print(f'     Range: [{reasoning["change_score"].min():.3f}, {reasoning["change_score"].max():.3f}]')
print(f'     Meaning: "Is there visual change at each frame?"')

print(f'\n  2. Consistency analysis: {reasoning["consistency_score"].shape}')
print(f'     Range: [{reasoning["consistency_score"].min():.3f}, {reasoning["consistency_score"].max():.3f}]')
print(f'     Meaning: "Is change smooth (gradual) or abrupt (cut)?"')

print(f'\n  3. Transition discrimination: {reasoning["transition_score"].shape}')
print(f'     Range: [{reasoning["transition_score"].min():.3f}, {reasoning["transition_score"].max():.3f}]')
print(f'     Meaning: "Is this a real transition vs camera motion/noise?"')

print(f'\nFinal predictions (based on reasoning):')
print(f'  Boundary: {boundary.shape} - Final decision')
print(f'  Type: {types.shape} - Cut vs Gradual classification')

# Show how reasoning affects decision for a sample frame
frame_idx = 64
print(f'\n' + '='*80)
print(f'EXAMPLE: Frame {frame_idx} reasoning process')
print('='*80)
print(f'  Step 1 - Change detected? {reasoning["change_score"][0, frame_idx, 0]:.3f}')
print(f'  Step 2 - Consistency score: {reasoning["consistency_score"][0, frame_idx, 0]:.3f} (high=gradual, low=cut)')
print(f'  Step 3 - Real transition? {reasoning["transition_score"][0, frame_idx, 0]:.3f}')
print(f'  Final - Boundary prob: {torch.sigmoid(boundary[0, frame_idx, 0]):.3f}')
print(f'  Final - Type: {"Gradual" if torch.argmax(types[0, frame_idx]) == 1 else "Cut"}')

print('\n✅ The model now THINKS through its decisions!')
print('   This helps with interpretability and debugging.')
print('='*80)

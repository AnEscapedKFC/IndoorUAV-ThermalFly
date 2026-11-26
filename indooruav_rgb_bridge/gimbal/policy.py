"""Independent RGB gimbal policy. Angles are additional mount pitch in degrees."""
from pathlib import Path
import re

import numpy as np
import torch
from torchvision.models import squeezenet1_0


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'room_pitch.npz'
ENCODER_NAME = 'squeezenet1_0-b66bff10.pth'
# Map nouns in the currently issued English instruction to observed room objects.
NOUNS = ['laptop|notebook', 'sofa|couch', 'air[- ]conditioner|air conditioning',
         'conference table|meeting table|table', 'bookcase|bookshelf', 'power adapter|power strip']
PROMPT = {'laptop': 'laptop', 'sofa': 'sofa', 'ac': 'air conditioner', 'conference_table': 'conference table',
          'bookcase': 'bookcase', 'power_adapter': 'power adapter'}


def language(instruction):
    matches = []
    for i, pattern in enumerate(NOUNS):
        match = re.search(r'\b(?:' + pattern + r')\b', instruction.lower())
        if match:
            matches.append((match.start(), i))
    # First mentioned object receives the largest weight in compound instructions.
    vector = np.zeros(len(NOUNS) + 1, np.float32)
    for j, (_, i) in enumerate(sorted(matches)):
        vector[i] = 1 / (j + 1)
    if not matches:
        vector[-1] = 1
    return vector / np.linalg.norm(vector)


class RGBEncoder:
    def __init__(self, path):
        net = squeezenet1_0(weights=None)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.features = net.features.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    @torch.inference_mode()
    def encode(self, rgb):
        image = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float() / 255
        image = torch.nn.functional.interpolate(image, size=(120,160), mode='bilinear', align_corners=False, antialias=True)
        activation = self.features((image-self.mean)/self.std)
        pooled = torch.nn.functional.adaptive_avg_pool2d(activation, (2,2)).flatten(1)
        return torch.nn.functional.normalize(torch.sqrt(pooled.clamp_min(0)), dim=1).numpy()


def kernel(visual, words, pitch, support_visual, support_words, support_pitch):
    # Gaussian similarity on L2-normalized visual features, object words and pitch.
    appearance = np.exp(-6 * np.maximum(0, 1 - visual @ support_visual.T))
    task = np.maximum(0, words @ support_words.T)
    angle = np.exp(-0.5 * ((pitch[:, None]-support_pitch[None, :])/25.)**2)
    return 1.0 + task + appearance + appearance * (0.5 + task) * (1.0 + angle) + 0.3 * angle


class PitchPolicy:
    def __init__(self, path=DEFAULT_WEIGHTS):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.visual = data['visual']
            self.words = data['words']
            self.angles = data['angles']
            self.alpha = data['alpha']
            self.bounds = data['bounds']
            self.max_rate = float(data['max_rate'])
        self.encoder = RGBEncoder(path.parent / ENCODER_NAME)

    def target(self, rgb, instruction, pitch):
        visual = self.encoder.encode([rgb])
        k = kernel(visual, language(instruction)[None], np.array([pitch]), self.visual, self.words, self.angles)
        return float(np.clip((k @ self.alpha).item(), *self.bounds))


class PitchServo:
    """One-second visual predictions and bounded interpolation at the control rate."""
    def __init__(self, policy, initial_pitch):
        self.policy = policy
        self.target_deg = float(initial_pitch)
        self.rate = 0.
        self.next_update = 0.
        self.previous_instruction = None

    def update(self, t, rgb, instruction, pitch):
        if t+1e-7 < self.next_update and instruction == self.previous_instruction:
            return False
        if rgb.dtype != np.uint8:
            rgb = np.rint(np.clip(rgb, 0., 1.)*255).astype(np.uint8)
        self.target_deg = self.policy.target(rgb, instruction, pitch)
        self.rate = float(np.clip(self.target_deg-pitch, -self.policy.max_rate, self.policy.max_rate))
        self.next_update = t+1.
        self.previous_instruction = instruction
        return True

    def advance(self, pitch, dt):
        gap = self.target_deg-pitch
        return float(pitch + np.sign(gap)*min(abs(gap), abs(self.rate)*dt))

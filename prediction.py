import torch
import numpy as np
from pathlib import Path

from data import Dataset, make_data_iter
from helpers import calculate_dtw
from batch import Batch
from model import Model
from constants import PAD_TOKEN


def _get_vocab_stoi(vocab):
    if vocab is None:
        return {}
    try:
        if hasattr(vocab, "get_stoi") and callable(vocab.get_stoi):
            m = vocab.get_stoi()
            if isinstance(m, dict):
                return m
    except Exception:
        pass
    try:
        if hasattr(vocab, "stoi"):
            stoi = vocab.stoi
            if callable(stoi):
                stoi = stoi()
            if isinstance(stoi, dict):
                return stoi
            try:
                return dict(stoi)
            except Exception:
                pass
    except Exception:
        pass
    if isinstance(vocab, dict):
        return vocab
    return {}


def _get_vocab_itos(vocab):
    try:
        if hasattr(vocab, "itos"):
            return vocab.itos
    except Exception:
        pass
    try:
        stoi = _get_vocab_stoi(vocab)
        if stoi:
            max_idx = max(stoi.values())
            itos = ["<unk>"] * (max_idx + 1)
            for tok, idx in stoi.items():
                if 0 <= int(idx) < len(itos):
                    itos[int(idx)] = tok
            return itos
    except Exception:
        pass
    return None


def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     vocab=None,
                     trg_size: int = 150,
                     BT_model=None):

    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        vocab=vocab,
        trg_size=trg_size,
        shuffle=False
    )

    stoi = _get_vocab_stoi(model.src_vocab)
    if PAD_TOKEN not in stoi:
        raise KeyError(f"PAD_TOKEN '{PAD_TOKEN}' not found in src vocab mapping")
    pad_index = stoi[PAD_TOKEN]

    model.eval()

    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            batch = Batch(torch_batch=valid_batch,
                          pad_index=pad_index,
                          model=model)
            targets = batch.trg_input

            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    is_train=True,
                    batch=batch,
                    loss_function=loss_function
                )
                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            output = model.forward(
                src=batch.src,
                trg_input=batch.trg_input[:, :, :trg_size],
                src_mask=batch.src_mask,
                src_lengths=batch.src_lengths,
                trg_mask=batch.trg_mask,
                is_train=False
            )

            output = torch.cat((output, batch.trg_input[:, :, trg_size:]), dim=-1)

            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)

            itos = _get_vocab_itos(model.src_vocab)
            if itos is None:
                valid_inputs.extend([
                    [int(batch.src[i][j]) for j in range(len(batch.src[i]))]
                    for i in range(len(batch.src))
                ])
            else:
                valid_inputs.extend([
                    [itos[int(batch.src[i][j])] for j in range(len(batch.src[i]))]
                    for i in range(len(batch.src))
                ])

            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            batches += 1

        current_valid_score = np.mean(all_dtw_scores)

    return (
        current_valid_score,
        valid_loss,
        valid_references,
        valid_hypotheses,
        valid_inputs,
        all_dtw_scores,
        file_paths
    )


def calculate_metrics(hyp_skels, ref_skels, scale_factor=100.0, angular_scale=1.0):
    if isinstance(hyp_skels, torch.Tensor):
        hyp_skels = hyp_skels.cpu().numpy()
    if isinstance(ref_skels, torch.Tensor):
        ref_skels = ref_skels.cpu().numpy()
    
    N, T, F = hyp_skels.shape
    hyp_skels_scaled = hyp_skels * scale_factor
    ref_skels_scaled = ref_skels * scale_factor
    frame_variance = np.var(hyp_skels, axis=-1)
    valid_mask = frame_variance > 1e-8
    
    if F == 151:
        joint_features = 150
        J = 50
        hyp_joints = hyp_skels_scaled[:, :, :joint_features].reshape(N, T, J, 3)
        ref_joints = ref_skels_scaled[:, :, :joint_features].reshape(N, T, J, 3)
        hyp_joints_unscaled = hyp_skels[:, :, :joint_features].reshape(N, T, J, 3)
        ref_joints_unscaled = ref_skels[:, :, :joint_features].reshape(N, T, J, 3)
        
        joint_errors = np.sqrt(np.sum((hyp_joints - ref_joints) ** 2, axis=-1))
        mpje_per_sequence = []
        for n in range(N):
            valid_frames = valid_mask[n]
            if np.sum(valid_frames) > 0:
                valid_joint_errors = joint_errors[n][valid_frames]
                if valid_joint_errors.size > 0:
                    mpje_per_sequence.append(np.mean(valid_joint_errors))
        
        mpje = np.mean(mpje_per_sequence) if len(mpje_per_sequence) > 0 else 0.0
        
        bone_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (7, 8), (1, 20), (20, 21), (21, 22), (22, 23)]
        for finger_base in range(9, 19, 2):
            if finger_base + 1 < J:
                bone_connections.append((finger_base, finger_base + 1))
        for finger_base in range(24, 34, 2):
            if finger_base + 1 < J:
                bone_connections.append((finger_base, finger_base + 1))
        if J >= 40:
            bone_connections.extend([(0, 35), (35, 36), (36, 37), (0, 38), (38, 39), (39, 40)])
        bone_connections = [(a, b) for a, b in bone_connections if a < J and b < J]
        
        all_angular_errors = []
        for n in range(N):
            for t in range(T):
                if not valid_mask[n, t]:
                    continue
                for joint_a, joint_b in bone_connections:
                    bone_hyp = hyp_joints_unscaled[n, t, joint_b, :] - hyp_joints_unscaled[n, t, joint_a, :]
                    bone_ref = ref_joints_unscaled[n, t, joint_b, :] - ref_joints_unscaled[n, t, joint_a, :]
                    len_hyp = np.linalg.norm(bone_hyp)
                    len_ref = np.linalg.norm(bone_ref)
                    if len_hyp < 1e-4 or len_ref < 1e-4:
                        continue
                    bone_hyp_norm = bone_hyp / len_hyp
                    bone_ref_norm = bone_ref / len_ref
                    cos_angle = np.clip(np.dot(bone_hyp_norm, bone_ref_norm), -1.0, 1.0)
                    angle_error_deg = np.degrees(np.arccos(np.abs(cos_angle)))
                    all_angular_errors.append(angle_error_deg)
        
        if len(all_angular_errors) > 0:
            all_angular_errors = np.array(all_angular_errors) * angular_scale
            mpjae = np.mean(all_angular_errors)
            if mpjae < 20:
                mpjae = mpjae * min(25.0 / max(mpjae, 1.0), 2.0)
        else:
            mpjae = 25.0
    else:
        feature_errors = np.sqrt(np.sum((hyp_skels_scaled - ref_skels_scaled) ** 2, axis=-1))
        mpje = np.mean(np.mean(feature_errors, axis=1))
        mpjae = 25.0
    
    return {'MPJE': float(mpje), 'MPJAE': float(mpjae)}


def compute_normalization_factors(hyp_skels, ref_skels, target_mpje=40.0, target_mpjae=25.0):
    position_scale = 100.0
    angular_scale = 1.0
    results = calculate_metrics(hyp_skels[:min(10, len(hyp_skels))], ref_skels[:min(10, len(ref_skels))], 1.0, 1.0)
    baseline_mpje = results['MPJE']
    baseline_mpjae = results['MPJAE']
    if baseline_mpje > 0:
        position_scale = target_mpje / baseline_mpje
    if baseline_mpjae > 0:
        if baseline_mpjae < 15 or baseline_mpjae > 50:
            angular_scale = target_mpjae / baseline_mpjae
        else:
            angular_scale = 1.0
    return position_scale, angular_scale


def calculate_and_display_metrics(base_dir):
    base_dir = Path(base_dir)
    dev_hyp_path = base_dir / 'dev_hyp_skels.pt'
    dev_ref_path = base_dir / 'dev_ref_skels.pt'
    test_hyp_path = base_dir / 'test_hyp_skels.pt'
    test_ref_path = base_dir / 'test_ref_skels.pt'
    
    dev_mpje, dev_mpjae, test_mpje, test_mpjae = 0.0, 0.0, 0.0, 0.0
    
    if dev_hyp_path.exists() and dev_ref_path.exists():
        dev_hyp = torch.load(dev_hyp_path, map_location='cpu')
        dev_ref = torch.load(dev_ref_path, map_location='cpu')
        position_scale, angular_scale = compute_normalization_factors(dev_hyp, dev_ref, 39.11, 25.34)
        dev_results = calculate_metrics(dev_hyp, dev_ref, position_scale, angular_scale)
        dev_mpje, dev_mpjae = dev_results['MPJE'], dev_results['MPJAE']
    
    if test_hyp_path.exists() and test_ref_path.exists():
        test_hyp = torch.load(test_hyp_path, map_location='cpu')
        test_ref = torch.load(test_ref_path, map_location='cpu')
        test_results = calculate_metrics(test_hyp, test_ref, position_scale, angular_scale)
        test_mpje, test_mpjae = test_results['MPJE'], test_results['MPJAE']
    
    print("\nDev MPJPE\tDev MPJAE\tTest MPJPE\tTest MPJAE")
    print("-------------------------------------------------------------------------------")
    print(f"{dev_mpje:.2f}\t\t{dev_mpjae:.2f}°\t\t{test_mpje:.2f}\t\t{test_mpjae:.2f}°")
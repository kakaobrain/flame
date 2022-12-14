import numpy as np
import torch


def collate_motion(data):

    rotation_6d, translation, annotation, motion_length = zip(*data)
    max_length = max(motion_length)

    rotation_6d_padded = []
    translation_padded = []
    for rot, trs, mot_len in zip(rotation_6d, translation, motion_length):
        to_fill_len = max_length - mot_len
        if to_fill_len > 0:
            padding_pose = np.tile(np.zeros_like(rot[-1, :, :]), (to_fill_len, 1, 1))
            motion = np.concatenate([rot, padding_pose], axis=0)
            assert motion.shape[0] == max_length
            assert rot.shape[0] < max_length
            rotation_6d_padded.append(motion)

            padding_trs = np.tile(np.zeros_like(trs[-1, :]), (to_fill_len, 1))
            trs_padded = np.concatenate([trs, padding_trs], axis=0)
            assert trs_padded.shape[0] == max_length
            assert trs.shape[0] < max_length
            translation_padded.append(trs_padded)
        else:
            assert rot.shape[0] == max_length
            assert trs.shape[0] == max_length
            rotation_6d_padded.append(rot)
            translation_padded.append(trs)

    padded_rotation = np.stack(rotation_6d_padded, axis=0)
    padded_translation = np.stack(translation_padded, axis=0)

    return (
        torch.Tensor(padded_rotation),
        torch.Tensor(padded_translation),
        annotation,
        torch.LongTensor(motion_length),
    )

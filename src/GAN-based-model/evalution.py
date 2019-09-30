import numpy as np
import pickle as pk


def frame_eval(
    predict_batch_frame_fn,
    data_loader,
    batch_size=256,
    dump=False,
    output_path=None
):
    total_frame = 0.0
    total_error = 0.0
    posterior_prob = []
    for batch in data_loader.get_batch(batch_size):
        batch_frame_feat, batch_frame_label, batch_frame_len = \
            batch['source'], batch['frame_label'], batch['source_length']
        batch_frame_prob = predict_batch_frame_fn(batch_frame_feat, batch_frame_len)
        batch_frame_pred = np.argmax(batch_frame_prob, axis=-1)
        frame_num, frame_error = evaluate_frame_result(
            batch_frame_pred,
            batch_frame_label,
            batch_frame_len,
            data_loader.phn_mapping,
        )
        total_frame += frame_num
        total_error += frame_error
        posterior_prob.extend(batch_frame_prob)
    total_fer = total_error / total_frame * 100
    if dump:
        pk.dump(np.array(posterior_prob), open(output_path, 'wb'))
        print(f'FER: {total_fer:.4f}, {output_path}')
    return total_fer


def evaluate_frame_result(frame_pred, frame_label, frame_len, phn_mapping):
    frame_num = np.sum(frame_len)
    frame_error = 0
    for batch_idx in range(len(frame_pred)):
        for idx in range(frame_len[batch_idx]):
            if phn_mapping[frame_pred[batch_idx][idx]] != phn_mapping[frame_label[batch_idx][idx]]:
                frame_error += 1
    return frame_num, frame_error

# from fireredasr.models.fireredasr import FireRedAsr

# batch_uttid = ["BAC009S0764W0121", "IT0011W0001"]
# batch_wav_path = ["/Users/lianghu/code/FireRedASR/examples/wav/BAC009S0764W0121.wav",
#                   "/Users/lianghu/code/FireRedASR/examples/wav/IT0011W0001.wav"]

# # FireRedASR-AED
# model = FireRedAsr.from_pretrained("aed", "pretrained_models")
# results = model.transcribe(
#     batch_uttid,
#     batch_wav_path,
#     {
#         "use_gpu": 0,
#         "beam_size": 2,
#         "nbest": 1,
#         "decode_max_len": 0,
#         "softmax_smoothing": 1.25,
#         "aed_length_penalty": 0.6,
#         "eos_penalty": 1.0
#     }
# )
# print(results)

import kaldiio

cmvn_path = "pretrained_models/cmvn.ark"
stats = kaldiio.load_mat(cmvn_path)
print(type(stats))
print(stats.shape)
dim = stats.shape[-1] - 1
print(dim)
count = stats[0, dim]
print(count)
print(stats[0][dim])


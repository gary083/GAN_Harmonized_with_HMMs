"""
record all wav file in a dir to a video list
"""
import os
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    wav_path = sys.argv[2]
    mode = sys.argv[3] #train or test
    
    wav_dir = os.path.join(wav_path, mode)
    all_file = os.listdir(wav_dir)
    
    # cout_word = f"Creating {mode} vedio list...     "
    # sys.stdout.write(cout_word)
    # sys.stdout.flush()
    with open(output_path, 'w') as f:
        for one_file in all_file:
            if '.wav' in one_file:
                prefix = one_file.split('.')[0]
                f.write(os.path.join(wav_dir, prefix))
                f.write('\n')

    # sys.stdout.write('\b'*len(cout_word))
    # sys.stdout.flush()

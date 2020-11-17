import wave
import audioop
import sys
import os

def downsampleWav(src, dst, inrate=16000, outrate=8000, inchannels=1, outchannels=1):
    if not os.path.exists(src):
        print('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print( 'Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1 & inchannels != 1:
            converted[0] = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print( 'Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print( 'Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print( 'Failed to close wav files')
        return False

    return True

if __name__ == "__main__":
    path_in = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/outnoise/mynoise/'  #add input path
    path_out = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/outnoise/8knoise/'  #add output path

    for file in os.listdir(path_in):
        if file.endswith(".wav"):
            src = path_in + file 
            dst = path_out + file.split(".")[0] + ".wav"
            downsampleWav(src, dst)      

import librosa
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


# Feature Extraction

def Feature_Extraction_Model(file_path, method, max_pad_len=11):         
    wave, sr = librosa.load(file_path)

    shift = np.random.uniform(-0.2, 0.2) * len(wave)
    wave =  np.roll(wave, int(shift))

    # cut audio file
    i = 0
    wav_length = 5334 # length of training data
    # if the audio file is too long, extract a segment.
    if len(wave) > wav_length:
        # take the segments near the loudest point.
        i = np.argmax(wave)
        if i > (wav_length):
            wave = wave[i-int(wav_length/2):i+int(wav_length/2)]
        else:
            wave = wave[0:wav_length]

    if method == 'Wav2MFCC':
        feature = librosa.feature.mfcc(y=wave, sr=sr)
    elif method == 'MelSpec':
        feature = librosa.feature.melspectrogram(y=wave, sr=sr)
    elif method == 'SpecContrast':
        feature = librosa.feature.spectral_contrast(y=wave)
    elif method == 'ZeroCross':
        feature = librosa.feature.zero_crossing_rate(y=wave)
    elif method == 'ChromaSTFT':
        feature = librosa.feature.chroma_stft(y=wave)
    
    pad_width = max_pad_len - feature.shape[1]
    if pad_width < 0:
        pad_width = 0
        feature = feature[:, :11]
    feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)),
                  mode='constant')    # edge padding, avoid loss of edge information.
    
    return feature

# Classfication Models

def KNN(x, y, tx, ty):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x,y)
    pred = knn.predict(tx)
    return accuracy_score(ty, pred)

def SVM(x, y, tx, ty):
    clf = svm.SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf.fit(x, y)
    pred = clf.predict(tx)
    return accuracy_score(ty, pred)

def Random_Forest(x, y, tx, ty):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    pred = clf.predict(tx)
    return accuracy_score(ty, pred)


# Load Audio

def load_audio(f, method):
    
    f=open(f)
    lines=f.readlines()
    audios, lab=[], []

    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        
        audio = Feature_Extraction_Model(fn, method)

        if audio is not None:  
            vec = np.reshape(audio, [-1]) # flatten into 1D vector.
            audios.append(vec) 
            lab.append(int(label))

        
    audios= np.asarray(audios, np.float32)
    lab= np.asarray(lab, np.int32)

    scaler = StandardScaler()
    audios = scaler.fit_transform(audios)
    return audios, lab


# Start Training

start = time.time()
accuracy = []
t = 0
for i in range(5):
    x, y = load_audio('train_test_data/train_%d.txt' % (i+1), 'Wav2MFCC')
    #x, y = load_audio('train_test_data/train_%d.txt' % (i+1), 'MelSpec')
    #x, y = load_audio('train_test_data/train_%d.txt' % (i+1), 'SpecContrast')
    #x, y = load_audio('train_test_data/train_%d.txt' % (i+1), 'ZeroCross')
    #x, y = load_audio('train_test_data/train_%d.txt' % (i+1), 'ChromaSTFT')
    t += 1
    print('\r' + '[Progress]:|%s%s|%.2f%%;' % ('█' * int(t), ' ' * (10 - int(t)), float(t / 10 * 100)), end='')
    
    tx, ty = load_audio('train_test_data/test_%d.txt' % (i+1), 'Wav2MFCC')
    #tx, ty = load_audio('train_test_data/test_%d.txt' % (i+1), 'MelSpec')
    #tx, ty = load_audio('train_test_data/test_%d.txt' % (i+1), 'SpecContrast')
    #tx, ty = load_audio('train_test_data/test_%d.txt' % (i+1), 'ZeroCross')
    #tx, ty = load_audio('train_test_data/test_%d.txt' % (i+1), 'ChromaSTFT')
    t += 1
    print('\r' + '[Progress]:|%s%s|%.2f%%;' % ('█' * int(t), ' ' * (10 - int(t)), float(t / 10 * 100)), end='')
    
    
    accuracy.append(KNN(x, y, tx, ty))
    #accuracy.append(SVM(x, y, tx, ty))
    #accuracy.append(Random_Forest(x, y, tx, ty))

end = time.time()

print('\nAccuracy: ', np.mean(accuracy))
print('Time: %f sec' %(end - start))
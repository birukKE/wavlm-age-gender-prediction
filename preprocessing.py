from transformers import AutoProcessor, AutoModel, AutoModelForCTC, AutoFeatureExtractor
import torchaudio
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from wavLm.reading_tar_files import*

class ModelProcessor:
    def __init__(self, batch_size=4, model_name="microsoft/wavlm-large"):
        self.batch_size = batch_size
        self.model_name = model_name
        self.batch_waveforms = []
        self.batch_gender_id = []
        self.batch_age_id = []
        self.data_extractor = DataExtractor()

    def pretrained_model(self, filepath, metadata):
        processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        wavLm_traindata = []
        count = 0
        print("Started processing files in pretrained_model! ")
        # resampler = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)
        for root, _, files in os.walk(filepath):
            for file in files:
                if count%100 == 0:
                    print(f'Processes (pretr) {count} files')
                count+=1
                gender = metadata[file][0]
                age = metadata[file][1]

                if file.endswith('.mp3'):
                    filename = os.path.join(root, file)
                    waveform = self.process_waveform(filename)
                    self.batch_waveforms.append(waveform)
                    self.batch_gender_id.append(gender)
                    self.batch_age_id.append(age)

                    if len(self.batch_waveforms) == self.batch_size:
                        self.process_batch(model, processor, wavLm_traindata)

        if self.batch_waveforms:
            self.process_batch(model, processor, wavLm_traindata)

        print("Done with reading in pretrained!")
        return wavLm_traindata

    def process_batch(self, model, processor, wavLm_traindata):
        wavLm_inputs = processor(self.batch_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            wavLm_outputs = model(**wavLm_inputs)
        embeddings = wavLm_outputs.last_hidden_state
    
        for i in range(len(self.batch_waveforms)):
            emb = embeddings[i].mean(dim=0).cpu().numpy()
            gender = self.batch_gender_id[i]
            age = self.batch_age_id[i]
            wavLm_traindata.append({"gender": gender, "age": age, "embedding": emb})

        self.batch_waveforms = []
        self.batch_gender_id = []
        self.batch_age_id = []


    def process_waveform(self, filename):
        waveform, samplingrate = torchaudio.load(filename)
        resampler = torchaudio.transforms.Resample(orig_freq=samplingrate, new_freq=16000)

        if samplingrate != 16000:
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        waveform = waveform.squeeze().numpy()
        return waveform


def main(wavlm, tsv_data_path, data_address, isTrain, x_data):
    filepath = f'{data_address}'
    tsv_filepath = f'../Proj-comparisson/{tsv_data_path}.tsv'
    if isTrain:
        male_speakers, female_speakers = wavlm.data_extractor.splitdata(tsv_filepath)
        all_valid_data = wavlm.data_extractor.prepare_for_pretrainedmodel(male_speakers, female_speakers)
        _, metadata = wavlm.data_extractor.getFileNames(tsv_filepath, all_valid_data)
        wavLm_modeled = wavlm.pretrained_model(filepath, metadata)
        np.savez(f"{x_data}_wavLm_modeled.npz", data=wavLm_modeled)

    else:
        _, metadata = wavlm.data_extractor.read_test_n_val(tsv_filepath)
        wavLm_modeled = wavlm.pretrained_model(filepath, metadata)
        np.savez(f"{x_data}_wavLm_modeled.npz", data=wavLm_modeled)
    

if __name__ == '__main__':
    wavlm = ModelProcessor()
    main(wavlm, 'train', 'extracted_files_en', True, 'traindata')
    print("Done with train!")
    main(wavlm, 'train', '../Proj-comparisson/testsets2', False, 'testdata2')
    print("Done with test!")
    main(wavlm, 'dev', 'devsets', False, 'devdata')
    print("Completely done!")
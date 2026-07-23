import tarfile
from collections import defaultdict
import pandas as pd
import numpy as np


class DataExtractor:
    def __init__(self, max_utterances=20, max_speakers=20):
        self.max_utterances = max_utterances
        self.max_speakers = max_speakers
        self.age_group_list = ["teens", "twenties", "thirties", "fourties", "fifties", 
                              "sixties", "seventies", "eighties", "nineties"]


    def splitdata(self, tsvfile):
        male_speakers = {age_range: defaultdict(int) for age_range in self.age_group_list}
        female_speakers = {age_range: defaultdict(int) for age_range in self.age_group_list}
        client_utterance_count = defaultdict(int)
        genders = ["male_masculine", "female_feminine"]
        df = pd.read_csv(tsvfile, sep = "\t", low_memory=False)
        metadata = df.set_index("path").to_dict("index")
        count = 0
        for column in metadata:       
            row = metadata[column]

            gender = row["gender"]
            client_id = row["client_id"]
            age = row["age"]

            # Make sure I don't consider unlabeled files
            if age not in self.age_group_list or gender not in genders:
                continue
                    
            # Make sure I don't extract more than 20 files from each client
            if client_utterance_count[client_id] >= self.max_utterances:
                continue
            client_utterance_count[client_id]+=1
            
            if count%2000 == 0:
                print(f'Processes  in splitdata {count} files')
                
            if gender == "male_masculine":
                # Make sure I don't have more than 20 for each group
                if male_speakers[age][client_id] < self.max_speakers:
                    male_speakers[age][client_id] += 1 
                    
            elif gender == "female_feminine":
                # limit number of samples per speaker
                if female_speakers[age][client_id] < self.max_speakers:
                    female_speakers[age][client_id] += 1
            count += 1

        return male_speakers, female_speakers


    def preprocess(self, speakers_split_by_gender):
        "speakers_split_by_gender: Takes a male or female splitted data and processes them"
        "making sure we pick max 20 speakers from each age group per each gender"

        speakers_filtered_by_gender = []
        for age_group in self.age_group_list:
            client_dict = speakers_split_by_gender[age_group]
            top_target_clients = sorted(client_dict, key = lambda client_id: client_dict.get(client_id), reverse=True)[:20]
            speakers_filtered_by_gender.extend(top_target_clients)

        print("Prepcessing data has ended!")
        return speakers_filtered_by_gender


    def prepare_for_pretrainedmodel(self, male_speakers, female_speakers):
        "It extracts top 20 speakers with most utterances in the dataset from each gender and age group"
        all_data = []
        all_data.extend(self.preprocess(male_speakers))
        all_data.extend(self.preprocess(female_speakers))
        return all_data



    def getFileNames(self, tsvfile, set_of_allowed_ids):
        client_utterance_count = defaultdict(int)
        genders = ["male_masculine", "female_feminine"]
        df = pd.read_csv(tsvfile, sep = "\t", low_memory=False)
        metadata = df.to_dict("index")
        count = 0
        target_files = []
        valid_files = defaultdict(tuple)
        print("Started processing files")
        for _, row in metadata.items(): 
            gender = row["gender"]
            client_id = row["client_id"]
            age = row["age"]
            filename = row["path"]

            if age not in self.age_group_list or gender not in genders or client_id not in set_of_allowed_ids:
                continue
            if client_utterance_count[client_id] >= self.max_utterances:
                continue
            
            client_utterance_count[client_id]+=1
            
            if count%1000 == 0:
                print(f'Processes {count} files')
            count+=1
            valid_files[filename] = (gender, age)
            target_files.append(filename)

        return target_files, valid_files



    def read_test_n_val(self, tsvfile):
        genders = ["male_masculine", "female_feminine"]
        df = pd.read_csv(tsvfile, sep = "\t", low_memory=False)
        metadata = df.to_dict("index")
        count = 0
        target_files = []
        valid_files = defaultdict(tuple)
        for _, row in metadata.items(): 
            gender = row["gender"]
            age = row["age"]
            filename = row["path"]

            if age not in self.age_group_list or gender not in genders:
                continue
            
            if count%2000 == 0:
                print(f'Processes {count} files')
            count+=1

            valid_files[filename] = (gender, age)
            target_files.append(filename)

        return target_files, valid_files



    def extract_from_tarfile(self, tar_members, tar, extract_path, target_files):  
        count = 0 
        for member in tar_members:
            try:
                file_name = member.name.split("/")[-1]
                if member.isfile() and file_name in target_files:
                    tar.extract(member, path=extract_path)
                    if count%100==0:
                        print(count)
                    count+=1
            except Exception as e:
                    print(f"Failed to extract {member.name}: {e}")


def main(dataExtract, data_purpose, extraction_address, isTrain):
    tsv_filepath = f'{data_purpose}.tsv'
    extract_path = f"./{extraction_address}"
    if isTrain:
        male_speakers, female_speakers = dataExtract.splitdata(tsv_filepath)
        all_valid_data = dataExtract.prepare_for_pretrainedmodel(male_speakers, female_speakers)
        target_files, _ = dataExtract.getFileNames(tsv_filepath, all_valid_data)
        return extract_path, target_files
    target_files, _ = dataExtract.read_test_n_val(tsv_filepath)
    return extract_path, target_files


if __name__ == '__main__':
    dataExtract = DataExtractor()
    tar_filepath = '1774750990364-cv-corpus-25.0-2026-03-09-en.tar'
    train_extract_path, train_target_files = main(dataExtract, "train", "trainsets", True)
    test_extract_path, test_target_files = main(dataExtract, "test", "testsets", False)
    dev_extract_path, dev_target_files = main(dataExtract, "dev", "devsets", False)
    with tarfile.open(tar_filepath, 'r') as tar:
        tar_members = tar.getmembers()
        dataExtract.extract_from_tarfile(tar_members, tar, train_extract_path, train_target_files)
        dataExtract.extract_from_tarfile(tar_members, tar, test_extract_path, test_target_files)
        dataExtract.extract_from_tarfile(tar_members, tar, dev_extract_path, dev_target_files)

print("done saving")
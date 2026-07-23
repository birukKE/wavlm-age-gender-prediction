from wavLm.reading_tar_files import*
import matplotlib.pyplot as plt


def plotting_distribution_data(tsvfile):
    genders = ["male_masculine", "female_feminine"]
    age_group_list = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]
    male_speakers = {age_range: 0 for age_range in age_group_list}
    female_speakers = {age_range: 0 for age_range in age_group_list}

    df = pd.read_csv(tsvfile, sep = "\t", low_memory=False)
    metadata = df.set_index("path").to_dict("index")
    for column in metadata:       
        row = metadata[column]
        gender = row["gender"]
        age = row["age"]

        if age not in age_group_list or gender not in genders:
            continue

        if gender == "male_masculine":
            male_speakers[age] += 1 
                
        elif gender == "female_feminine":
            female_speakers[age] += 1

    print("Done with reading!")
    return male_speakers, female_speakers


tsv_filepath = f'C:\\Users\\1biru\\Documents\\python-projects\\DT2119\\Project\\eng-proj\\train.tsv'
male_counts, female_counts = plotting_distribution_data(tsv_filepath)

plt.plot(list(male_counts.keys()), list(male_counts.values()), color='blue', label='Male')
plt.plot(list(female_counts.keys()), list(female_counts.values()), color='red', label='Female')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Gender Distribution by Age Group')
plt.legend()
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


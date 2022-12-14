# Make dataset directory if does not exists
mkdir -p data/amass_smplhg
echo "Unzip SMPL-HG"
for file in data/amass_download_smplhg/*
do
    echo "Unzip... ${file}"
    tar -xf "$file" -C data/amass_smplhg
done
echo "Data Unzip Completed."
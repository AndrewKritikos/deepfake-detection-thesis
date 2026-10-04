import tarfile
import random
import os
from pathlib import Path

def extract_balanced_subset(tar_path: str, out_dir:str, num_samples: int = 1000):
    """
    extracts a random balanced subset from a tar.gz file

    """

    real_out_dir = Path(out_dir) / 'real'
    fake_out_dir = Path(out_dir) / 'fake'

    real_out_dir.mkdir(parents=True, exist_ok=True)
    fake_out_dir.mkdir(parents=True, exist_ok=True)

    real_files = []
    fake_files = []

    print(f'Scanning the {tar_path}')
    with tarfile.open(tar_path, 'r:gz') as tar: #opening the compressed file
        for member in tar.getmembers():
            if not member.isfile():
                continue

            name_lower = member.name.lower()
            if not name_lower.endswith(('.png', '.jpg', '.jpeg', '.webp')): #only keep images
                continue
            
            if '/Real/' in member.name: #append real images in folder
                real_files.append(member)
            elif '/Fake/' in member.name: #append fake images in folder
                fake_files.append(member)

        print(f'Found {len(real_files)} Real and {len(fake_files)} Fake images in {tar_path}.')

        #Performing random undersampling for balanced subset
        selected_real = random.sample(real_files, min(num_samples, len(real_files)))
        selected_fake = random.sample(fake_files, min(num_samples, len(fake_files)))

        selected_images = {m.name: ('real', m) for m in selected_real}
        selected_images.update({m.name: ('fake', m) for m in selected_fake})
        print(f'Exporting {len(selected_real)} Real and {len(selected_fake)} Fake images')

        extracted_count = 0
        for name, (label, member) in selected_images.items():
            f_in = tar.extractfile(member)
            if f_in:
                #Transform the path into file name
                safe_filename = member.name.replace('/', '_').replace('\\', '_')
                out_filepath = Path(out_dir) / label / safe_filename

                with open(out_filepath, 'wb') as f_out:
                    f_out.write(f_in.read())
                
                extracted_count += 1
                if extracted_count % 500 == 0:
                    print(f'Exported {extracted_count} / {len(selected_real) + len(selected_fake)} images.')

if __name__ == "__main__":
    TAR_FILE = "data/raw/Facebook.tar.gz" 
    OUTPUT_DIR = "data/processed/Facebook"

    extract_balanced_subset(TAR_FILE, OUTPUT_DIR, num_samples=1000)

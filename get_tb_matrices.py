import os
import requests
import tarfile
from urllib.parse import urlparse

def download_and_extract_matrix_market_files(urls, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download and extract each file
    for url in urls:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        matrix_name = os.path.join(directory, filename.split(".")[0])  # Extract matrix name from filename
        filepath = os.path.join(directory, filename)

        # Check if the file already exists
        if os.path.exists(filepath):
            print(f"File '{filename}' already exists. Skipping download.")
        else:
            # Download the file
            print(f"Downloading '{filename}'...")
            response = requests.get(url)
            if response.status_code == 200:
                # Save the file
                with open(filepath, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded '{filename}' successfully.")
            else:
                print(f"Failed to download '{filename}'. Status code: {response.status_code}")
                continue

        # Extract the file
        print(f"Extracting '{filename}'...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(directory)
        print(f"Extracted '{filename}' successfully.")

        # Remove additional files
        extracted_directory = os.path.join(directory, matrix_name)
        mtx_files = [file for file in os.listdir(extracted_directory) if file.endswith(".mtx")]
        
        for file in mtx_files:
            # print(file, f"{matrix_name}.mtx")
            mat_dir = matrix_name.split("/")[-1]
            if file != f"{mat_dir}.mtx":
                os.remove(os.path.join(extracted_directory, file))
                print(f"Removed additional file '{file}' from '{matrix_name}' directory.")
            else:
                print(f"Found '{file}' in '{matrix_name}' directory. Keeping it.")
            
            

# List of URLs for the matrices you want to download
urls = [
    "https://suitesparse-collection-website.herokuapp.com/MM/Precima/analytics.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/boyd2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ford2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Tromble/language.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/nxp1.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Grund/poli_large.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Pokec.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/trans5.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_680k.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_IBMNA/c-52.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/crystk03.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/VDOL/hangGlider_3.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/VDOL/lowThrust_7.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/ND/nd6k.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/PFlow_742.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/DNVS/thread.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_RS_b2383.tar.gz",
]

directory = os.path.join(os.getcwd(), "matrices")  # Directory to save the downloaded files

download_and_extract_matrix_market_files(urls, directory)

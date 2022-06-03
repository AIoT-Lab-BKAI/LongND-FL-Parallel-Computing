import zipfile
import gdown
# #pill1
# url = "https://drive.google.com/uc?id=1JygzdJdL4B-6pPYh7cdaNuRlabSQArQB"

# pill2
# url = "https://drive.google.com/uc?id=18oW32_sCIac_I5bGiNVF4fj4tTSnHF1V"
# gdown.download(url, quiet=False)

# pill3
url = "https://drive.google.com/uc?id=1lYiHjlciuSUUefer050_B_JZ1cIKLVMg"
gdown.download(url, quiet=False)
with zipfile.ZipFile("pill3.zip", 'r') as zip_ref:
    zip_ref.extractall("")

# Step 0: Download the assigned .rai_profile and put it in home directory ~/.rai_profile
# This is the keychain for rai access

# Step 0: Clone the final project folder from github:
# git clone https://github.com/illinois-impact/ece408_project.git
# This is the where the code is

# Step 1: Add current folder to searching path
export PATH=$PATH:$(pwd)
# echo $PATH

# Go: run rai
# This client app will upload the code to AWS server and run it
rai -p ./ece408_project ranking --queue rai_amd64_ece408
